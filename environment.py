"""
environment.py
--------------
CASREnv: A Gymnasium-compatible RL environment wrapping the S-Cache simulation.

The environment replays Azure Functions invocation traces row by row.
At every RL step, it processes STEP_SIZE invocations, then asks the agent
to choose a scaling action.

STATE SPACE (21 dimensions = 7 features × 3 queues):
    Per queue (Q0, Q1, Q2):
        [0] cap             – Maximum containers allowed
        [1] current_len     – Containers currently cached
        [2] invocations     – Invocations processed in this step
        [3] cold_starts     – Cold starts in this step
        [4] evictions       – Containers evicted in this step
        [5] running         – Containers in cold-start phase
        [6] wmt             – Wasted memory time (container-ms)
    All features are normalized to [0, 1] using running statistics.

ACTION SPACE (27 discrete actions):
    3 queues × 3 choices per queue = 3^3 = 27 actions
    Each queue's choice: 0=scale down, 1=hold, 2=scale up
    Scaling ratio = ξ = 0.25 (CASR paper default)

STEP:
    1. Process STEP_SIZE invocations from the dataset.
    2. Accumulate S-Cache metrics.
    3. Decode action, apply to queues.
    4. Compute reward via RewardCalculator.
    5. Return (new_state, reward, terminated, truncated, info).
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

from scache import SCache
from reward import RewardCalculator, RewardWeights


# Number of invocations processed per RL step (δ from CASR paper)
# Smaller = more frequent decisions; larger = smoother but less responsive
STEP_SIZE = 500


class CASREnv(gym.Env):
    """
    RL environment for the CASR+ project.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed Azure Functions dataset.
    weights : RewardWeights
        Reward function weights (θ, α, β, γ).
    step_size : int
        Number of invocations per RL step.
    sla_threshold : float
        Cold start rate that triggers the SLA penalty.
    initial_caps : list[int] or None
        Initial queue capacities [Q0, Q1, Q2].
    """

    metadata = {"render_modes": []}

    # Action decoding: map integer 0-26 → (a0, a1, a2)
    # Each action component: 0=down, 1=hold, 2=up
    _ACTION_DECODE = [
        (a0, a1, a2)
        for a0 in range(3)
        for a1 in range(3)
        for a2 in range(3)
    ]  # 27 entries

    def __init__(
        self,
        data: pd.DataFrame,
        weights: Optional[RewardWeights] = None,
        step_size: int = STEP_SIZE,
        sla_threshold: float = 0.15,
        initial_caps: Optional[list] = None,
    ):
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.weights = weights or RewardWeights.default()
        self.step_size = step_size
        self.initial_caps = initial_caps or [200, 100, 50]

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(21,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(27)

        # Internal components
        self._scache = SCache(initial_caps=list(self.initial_caps))
        self._reward_calc = RewardCalculator(sla_threshold=sla_threshold)

        # Episode tracking
        self._cursor = 0           # Current row index in data
        self._prev_metrics = None
        self._prev_caps = None
        self._episode_reward = 0.0
        self._episode_cold_starts = 0
        self._episode_invocations = 0
        self._step_count = 0

        # State normalizer (online, per episode)
        self._state_mean = np.zeros(21, dtype=np.float32)
        self._state_std = np.ones(21, dtype=np.float32)
        self._state_count = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium interface
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._scache.reset(initial_caps=list(self.initial_caps))
        self._reward_calc.reset()
        self._cursor = 0
        self._prev_metrics = None
        self._prev_caps = list(self.initial_caps)
        self._episode_reward = 0.0
        self._episode_cold_starts = 0
        self._episode_invocations = 0
        self._step_count = 0
        self._state_count = 0
        self._state_mean = np.zeros(21, dtype=np.float32)
        self._state_std = np.ones(21, dtype=np.float32)

        # Shuffle data for each episode to avoid overfitting to order
        if options and options.get("shuffle", True):
            self.data = self.data.sample(frac=1).reset_index(drop=True)

        state = self._get_state()
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        One RL step:
          1. Process STEP_SIZE invocations.
          2. Apply scaling action.
          3. Compute reward.
          4. Return (obs, reward, terminated, truncated, info).
        """
        self._scache.reset_step_stats()

        # ── Process invocations ───────────────────────────────────────────────
        end_cursor = min(self._cursor + self.step_size, len(self.data))
        batch = self.data.iloc[self._cursor:end_cursor]
        self._cursor = end_cursor

        # Simulate elapsed time for WMT calculation
        # Each minute-row represents 60 000 ms
        time_delta_ms = len(batch) * 60_000.0 / max(1, self.step_size)

        for row in batch.itertuples(index=False):
            # row: func_id, day, arrival_minute, count, cold_start_ms, memory_mb
            # Process each invocation count times (capped for speed)
            invoc_count = min(int(row.count), 50)
            for _ in range(max(1, invoc_count)):
                self._scache.process_invocation(
                    func_id=str(row.func_id),
                    cold_start_ms=float(row.cold_start_ms),
                    memory_mb=float(row.memory_mb),
                )

        self._scache.accumulate_wmt(time_delta_ms)

        # ── Get metrics before action ─────────────────────────────────────────
        metrics = self._scache.get_step_metrics()
        current_caps = metrics["capacities"]

        # ── Apply action ──────────────────────────────────────────────────────
        action_tuple = self._ACTION_DECODE[int(action)]
        self._scache.apply_action(action_tuple)
        new_caps = [q.cap for q in self._scache.queues]

        # ── Compute reward ────────────────────────────────────────────────────
        reward, components = self._reward_calc.compute(
            metrics=metrics,
            prev_metrics=self._prev_metrics,
            weights=self.weights,
            prev_capacities=self._prev_caps,
            current_capacities=current_caps,
        )

        # ── Update episode stats ──────────────────────────────────────────────
        self._episode_reward += reward
        self._episode_cold_starts += metrics["total_cold_starts"]
        self._episode_invocations += metrics["total_invocations"]
        self._prev_metrics = metrics
        self._prev_caps = current_caps
        self._step_count += 1

        # ── Termination ───────────────────────────────────────────────────────
        terminated = False
        truncated = self._cursor >= len(self.data)

        # ── Next observation ──────────────────────────────────────────────────
        next_state = self._get_state()

        info = {
            **components,
            "action_tuple": action_tuple,
            "capacities": new_caps,
            "episode_cold_start_rate": (
                self._episode_cold_starts / max(1, self._episode_invocations)
            ),
            "step": self._step_count,
        }

        return next_state, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # State construction
    # ──────────────────────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """Build and normalise the 21-dim state vector."""
        raw = self._scache.get_state_vector()

        # Online normalisation: batch-mean / std approach
        # Update running statistics
        self._state_count += 1
        if self._state_count == 1:
            self._state_mean = raw.copy()
            self._state_std = np.ones(21, dtype=np.float32)
        else:
            delta = raw - self._state_mean
            self._state_mean += delta / self._state_count
            self._state_std = np.maximum(
                np.abs(raw - self._state_mean) * 0.1 + self._state_std * 0.9,
                1e-6,
            )

        normalised = (raw - self._state_mean) / (self._state_std + 1e-8)
        # Clip to reasonable range
        return np.clip(normalised, -5.0, 5.0).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Weight update (called by training loop when Meta-model gives new weights)
    # ──────────────────────────────────────────────────────────────────────────

    def update_weights(self, weights: RewardWeights):
        """Update reward weights mid-training (used by Meta-model predictions)."""
        self.weights = weights

    # ──────────────────────────────────────────────────────────────────────────
    # Episode summary
    # ──────────────────────────────────────────────────────────────────────────

    def episode_summary(self) -> dict:
        return {
            "total_reward": self._episode_reward,
            "cold_start_rate": (
                self._episode_cold_starts / max(1, self._episode_invocations)
            ),
            "total_cold_starts": self._episode_cold_starts,
            "total_invocations": self._episode_invocations,
            "steps": self._step_count,
        }
