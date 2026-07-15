"""
scache.py
---------
Implements S-Cache, the container keep-alive strategy from the CASR paper.

CASR classifies functions into 3 queues based on their cold-start overhead:
    Q0: 0 – 1 000 ms   (0–1 second)
    Q1: 1 000 – 60 000 ms  (1–60 seconds)
    Q2: > 60 000 ms    (> 60 seconds)

Each queue uses a W-TinyLFU inspired policy:
  • A small "window" cache (LRU) absorbs bursty new arrivals.
  • A larger "main" cache (LRU) stores frequently reused containers.
  • A Count-Min Sketch tracks access frequency for eviction decisions.
  • The Bloom Filter is approximated by the frequency sketch.

Key insight from CASR: containers that are currently RUNNING (loading
dependencies) cannot serve new requests, even though they exist in memory.
This is unlike regular caching.  Available containers = cached – running.

State tracked per queue (used by the RL environment):
    cap     Maximum number of containers allowed in this queue
    len     Current number of containers cached
    invoc   Invocations processed since last RL step
    cold    Cold starts since last RL step
    evicted Containers evicted since last RL step
    running Containers currently in cold-start phase
    wmt     Wasted Memory Time (container-seconds idle, since last step)
"""

from collections import OrderedDict, defaultdict
import numpy as np
from typing import Dict, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Count-Min Sketch (lightweight frequency approximator)
# ──────────────────────────────────────────────────────────────────────────────

class _CountMinSketch:
    """
    Approximates access frequency using a matrix of counters.
    Space: O(width * depth).  Error: O(n / width) with probability 1 – 2^-depth.
    """

    def __init__(self, width: int = 1024, depth: int = 4, seed: int = 0):
        self.width = width
        self.depth = depth
        rng = np.random.default_rng(seed)
        # Random hash seeds
        self._seeds = rng.integers(0, 2**31, size=depth)
        self._table = np.zeros((depth, width), dtype=np.int32)
        self._total = 0

    def _hash(self, key: str, row: int) -> int:
        h = hash(key) ^ int(self._seeds[row])
        return abs(h) % self.width

    def increment(self, key: str, amount: int = 1):
        for row in range(self.depth):
            col = self._hash(key, row)
            self._table[row, col] += amount
        self._total += amount
        # Periodic halving to keep counts fresh (freshness mechanism)
        if self._total > 100_000:
            self._table //= 2
            self._total //= 2

    def estimate(self, key: str) -> int:
        return int(min(
            self._table[row, self._hash(key, row)]
            for row in range(self.depth)
        ))

    def reset(self):
        self._table[:] = 0
        self._total = 0


# ──────────────────────────────────────────────────────────────────────────────
# Single S-Cache queue
# ──────────────────────────────────────────────────────────────────────────────

# Overhead bucket boundaries (milliseconds), matching CASR paper
BUCKET_BOUNDS = [(0, 1_000), (1_000, 60_000), (60_000, float("inf"))]


class SCacheQueue:
    """
    One queue of S-Cache.  Manages warm containers for functions whose
    cold-start overhead falls within this queue's overhead bucket.

    The structure mirrors W-TinyLFU:
        window_cache  (LRU, 20% of capacity)  ←  new containers arrive here
        main_cache    (LRU, 80% of capacity)  ←  promoted after surviving window
    """

    def __init__(self, queue_id: int, initial_cap: int = 50):
        self.queue_id = queue_id
        self._cap = max(1, initial_cap)

        # Window size = 20% of capacity (W-TinyLFU default)
        self._window_size = max(1, int(self._cap * 0.2))
        self._main_size = max(1, self._cap - self._window_size)

        # LRU structures: OrderedDict, most-recent at end
        self._window: OrderedDict[str, float] = OrderedDict()  # func_id → memory_mb
        self._main: OrderedDict[str, float] = OrderedDict()

        # Running containers (currently doing cold start; unavailable)
        self._running: Dict[str, float] = {}  # func_id → cold_start_ms

        self._freq = _CountMinSketch()

        # Per-step statistics (reset by SCache after each RL step)
        self.step_invocations = 0
        self.step_cold_starts = 0
        self.step_evictions = 0
        self.step_wmt = 0.0     # Wasted memory-time (container-ms idle)

        # Cumulative cold-start overhead (for R4 reward term)
        self.step_cold_start_overhead_ms = 0.0

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def cap(self) -> int:
        return self._cap

    @property
    def current_len(self) -> int:
        return len(self._window) + len(self._main)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def available_count(self) -> int:
        """Available = cached but NOT currently running."""
        in_cache = set(self._window) | set(self._main)
        running = set(self._running)
        return len(in_cache - running)

    # ── Lookup and insert ────────────────────────────────────────────────────

    def lookup(self, func_id: str, cold_start_ms: float, memory_mb: float) -> bool:
        """
        Process one function invocation.

        Returns True  → warm start (container reused).
        Returns False → cold start (new container needed).

        The cold_start_ms tells us how long this container will be
        'unavailable' (running) after a cold start.
        """
        self.step_invocations += 1
        self._freq.increment(func_id)

        # Check if there's an available (not running) container
        in_cache = (func_id in self._window) or (func_id in self._main)
        is_available = in_cache and (func_id not in self._running)

        if is_available:
            # Warm start: update LRU order
            self._touch(func_id)
            # Accumulate idle time for the container we just freed  →  NOT wasted
            return True  # warm start
        else:
            # Cold start
            self.step_cold_starts += 1
            self.step_cold_start_overhead_ms += cold_start_ms

            # Mark this container as running (unavailable during load)
            self._running[func_id] = cold_start_ms

            # Insert into window cache (evicting if full)
            evicted = self._insert_into_window(func_id, memory_mb)
            if evicted:
                self.step_evictions += len(evicted)

            return False  # cold start

    def complete_running(self, func_id: str):
        """
        Mark a running container as done loading.
        In simulation we call this immediately (zero execution time assumption).
        """
        self._running.pop(func_id, None)

    def _touch(self, func_id: str):
        """Move func_id to most-recently-used position."""
        if func_id in self._window:
            self._window.move_to_end(func_id)
        elif func_id in self._main:
            self._main.move_to_end(func_id)

    def _insert_into_window(self, func_id: str, memory_mb: float) -> list:
        """
        Insert a new container into the window cache.
        If window is full, try to promote the evicted item to main cache.
        Returns list of containers that were fully evicted from S-Cache.
        """
        evicted = []

        # Evict from window if full
        while len(self._window) >= self._window_size:
            oldest_id, oldest_mem = self._window.popitem(last=False)
            # Try to promote to main cache
            promotion_evictions = self._try_promote_to_main(oldest_id, oldest_mem)
            evicted.extend(promotion_evictions)

        self._window[func_id] = memory_mb
        return evicted

    def _try_promote_to_main(self, func_id: str, memory_mb: float) -> list:
        """
        Promote func_id to main cache.
        If main cache is full, compare frequencies and evict the loser.
        Returns list of truly evicted (removed from S-Cache entirely) items.
        """
        evicted = []

        if len(self._main) < self._main_size:
            self._main[func_id] = memory_mb
        else:
            # Find the victim in main cache (oldest / LRU)
            victim_id, victim_mem = next(iter(self._main.items()))
            freq_candidate = self._freq.estimate(func_id)
            freq_victim = self._freq.estimate(victim_id)

            if freq_candidate > freq_victim:
                # Candidate wins: evict the victim, insert candidate
                del self._main[victim_id]
                self._main[func_id] = memory_mb
                evicted.append(victim_id)
            else:
                # Victim stays; candidate is discarded
                evicted.append(func_id)

        return evicted

    def accumulate_wmt(self, time_delta_ms: float):
        """
        Add wasted memory time: every idle container accumulates WMT.
        Called once per RL step with the elapsed simulated time.
        """
        idle_count = self.current_len - self.running_count
        self.step_wmt += max(0, idle_count) * time_delta_ms

    # ── Scaling ──────────────────────────────────────────────────────────────

    def scale(self, ratio: float) -> list:
        """
        Scale capacity by (1 + ratio).  ratio > 0 = grow, < 0 = shrink.
        Returns list of evicted container IDs when shrinking.
        Minimum capacity is always 1.
        """
        new_cap = max(1, int(self._cap * (1.0 + ratio)))
        evicted = []

        if new_cap < self._cap:
            # Need to evict containers to fit new capacity
            new_window_size = max(1, int(new_cap * 0.2))
            new_main_size = max(1, new_cap - new_window_size)

            while len(self._main) > new_main_size:
                fid, _ = self._main.popitem(last=False)  # LRU eviction
                evicted.append(fid)
                self.step_evictions += 1

            while len(self._window) > new_window_size:
                fid, _ = self._window.popitem(last=False)
                evicted.append(fid)
                self.step_evictions += 1

            self._window_size = new_window_size
            self._main_size = new_main_size

        else:
            self._window_size = max(1, int(new_cap * 0.2))
            self._main_size = max(1, new_cap - self._window_size)

        self._cap = new_cap
        return evicted

    # ── Step reset ───────────────────────────────────────────────────────────

    def reset_step_stats(self):
        """Clear per-step counters at the start of each RL step."""
        self.step_invocations = 0
        self.step_cold_starts = 0
        self.step_evictions = 0
        self.step_wmt = 0.0
        self.step_cold_start_overhead_ms = 0.0

    def get_state_vector(self) -> np.ndarray:
        """
        Return the 7-dim state vector for this queue (used in CASREnv).
            [cap, current_len, invocations, cold_starts, evictions, running, wmt]
        """
        return np.array([
            self._cap,
            self.current_len,
            self.step_invocations,
            self.step_cold_starts,
            self.step_evictions,
            self.running_count,
            self.step_wmt,
        ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# S-Cache: three-queue manager
# ──────────────────────────────────────────────────────────────────────────────

class SCache:
    """
    Full S-Cache with 3 queues corresponding to CASR's overhead buckets.

    Q0: 0–1s       Q1: 1–60s       Q2: >60s

    On each function invocation:
      1. Route the function to the appropriate queue.
      2. Call queue.lookup() to get warm/cold result.
      3. If cold start, immediately call queue.complete_running() to
         release the container (simplification: we don't simulate execution time).
    """

    # CASR paper overhead bucket boundaries (in milliseconds)
    Q_BOUNDS = [(0, 1_000), (1_000, 60_000), (60_000, float("inf"))]
    # Initial queue capacities (tuned to match CASR paper setup)
    INITIAL_CAPS = [200, 100, 50]
    # Scaling ratio per RL action step (ξ = 0.25 from CASR paper)
    SCALE_RATIO = 0.25

    def __init__(self, initial_caps: Optional[list] = None):
        caps = initial_caps or self.INITIAL_CAPS
        self.queues = [SCacheQueue(i, cap) for i, cap in enumerate(caps)]

    def _route(self, cold_start_ms: float) -> int:
        """Return the queue index (0, 1, or 2) for this overhead value."""
        for i, (lo, hi) in enumerate(self.Q_BOUNDS):
            if lo <= cold_start_ms < hi:
                return i
        return 2  # fallback to Q2

    def process_invocation(
        self, func_id: str, cold_start_ms: float, memory_mb: float
    ) -> Tuple[bool, int]:
        """
        Process one function invocation.

        Returns (is_warm_start, queue_index).
        """
        q_idx = self._route(cold_start_ms)
        q = self.queues[q_idx]
        warm = q.lookup(func_id, cold_start_ms, memory_mb)
        # Immediately complete running state (simplified: no execution delay)
        if not warm:
            q.complete_running(func_id)
        return warm, q_idx

    def apply_action(self, action_tuple: Tuple[int, int, int]):
        """
        Apply an RL scaling action.
        action_tuple: (a0, a1, a2) where each ∈ {0=down, 1=hold, 2=up}.
        """
        for q_idx, a in enumerate(action_tuple):
            if a == 0:
                self.queues[q_idx].scale(-self.SCALE_RATIO)
            elif a == 2:
                self.queues[q_idx].scale(+self.SCALE_RATIO)
            # a == 1 → no change

    def accumulate_wmt(self, time_delta_ms: float):
        for q in self.queues:
            q.accumulate_wmt(time_delta_ms)

    def get_state_vector(self) -> np.ndarray:
        """
        Full 21-dim state vector = 7 features × 3 queues.
        Order: [Q0_features, Q1_features, Q2_features]
        """
        return np.concatenate([q.get_state_vector() for q in self.queues])

    def reset_step_stats(self):
        for q in self.queues:
            q.reset_step_stats()

    def get_step_metrics(self) -> dict:
        """Aggregate per-step metrics across all queues."""
        return {
            "total_invocations": sum(q.step_invocations for q in self.queues),
            "total_cold_starts": sum(q.step_cold_starts for q in self.queues),
            "total_evictions": sum(q.step_evictions for q in self.queues),
            "total_wmt": sum(q.step_wmt for q in self.queues),
            "total_cso": sum(q.step_cold_start_overhead_ms for q in self.queues),
            "cold_starts_per_queue": [q.step_cold_starts for q in self.queues],
            "wmt_per_queue": [q.step_wmt for q in self.queues],
            "cso_per_queue": [q.step_cold_start_overhead_ms for q in self.queues],
            "capacities": [q.cap for q in self.queues],
        }

    def reset(self, initial_caps: Optional[list] = None):
        """Full reset between episodes."""
        caps = initial_caps or self.INITIAL_CAPS
        for i, q in enumerate(self.queues):
            q._cap = max(1, caps[i])
            q._window_size = max(1, int(q._cap * 0.2))
            q._main_size = max(1, q._cap - q._window_size)
            q._window.clear()
            q._main.clear()
            q._running.clear()
            q._freq.reset()
            q.reset_step_stats()
