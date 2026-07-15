"""
reward.py
---------
Extended Reward Function for the CASR+ project.

CASR's original reward (Eq. 6 in the paper):
    Reward = -(θ × norm(R1) + (1-θ) × norm(R2))
    where R1 = total cold starts, R2 = ΔWMT.

Our extension adds three more penalty terms:

    R3  = Oscillation penalty
          Measures how much the agent changed queue capacities vs. the
          previous step.  Penalises rapid flip-flopping, which wastes
          resources on eviction/insertion cycles.

    R4  = Cold-Start Overhead (CSO) penalty  [key CASR metric]
          R4 = Σ_k (cold_start_overhead_k × cold_starts_k)
          This is CASR's CSO formula (Eq. 3 in the paper).
          A 60-second cold start is penalised 60× more than a 1-second one.
          This encourages the agent to prioritise keeping heavy functions warm.

    SLA = SLA violation penalty
          Smooth sigmoid penalty when cold_start_rate > sla_threshold.
          Using a smooth sigmoid (instead of a hard step function) provides
          a gradient signal even BEFORE the threshold is breached, helping
          the agent learn to stay safely below the SLA limit.

Full reward:
    Reward = -(θ·norm(R1) + (1-θ)·norm(R2))
             - α·norm(R3)
             - β·SLA_sigmoid
             - γ·norm(R4)

The four weights (θ, α, β, γ) are learned by the Meta-model and passed
in at each episode.  Defaults match CASR's θ=0.8 finding.

Why each weight matters:
    θ  –  Primary trade-off: cold starts vs memory.  0.8 = mostly care about latency.
    α  –  How much we punish erratic scaling.  Small α (0.1) = stability is secondary.
    β  –  How hard we hit SLA breaches.  Large β (0.5) = treat SLA as hard constraint.
    γ  –  How much we care about overhead-weighted cold starts.  0.2 = secondary goal.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Running statistics for normalization
# ──────────────────────────────────────────────────────────────────────────────

class _RunningNormalizer:
    """
    Tracks running min/max of a scalar for online normalization.
    Matches CASR paper's normalization approach (Eq. 6).
    """
    def __init__(self):
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update(self, val: float):
        self.min_val = min(self.min_val, val)
        self.max_val = max(self.max_val, val)

    def normalize(self, val: float) -> float:
        rng = self.max_val - self.min_val
        if rng < 1e-8:
            return 0.0
        return (val - self.min_val) / rng


# ──────────────────────────────────────────────────────────────────────────────
# Reward weights dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RewardWeights:
    """
    The four reward weights.  Either fixed or predicted by the Meta-model.

    theta  (θ): Cold start vs memory trade-off weight.  Range [0, 1].
    alpha  (α): Oscillation penalty weight.             Range [0, 1].
    beta   (β): SLA violation penalty weight.           Range [0, 1].
    gamma  (γ): CSO (overhead-weighted) penalty weight. Range [0, 1].
    """
    theta: float = 0.8
    alpha: float = 0.1
    beta: float = 0.3
    gamma: float = 0.2

    def as_array(self) -> np.ndarray:
        return np.array([self.theta, self.alpha, self.beta, self.gamma], dtype=np.float32)

    @classmethod
    def from_array(cls, arr) -> "RewardWeights":
        return cls(theta=float(arr[0]), alpha=float(arr[1]),
                   beta=float(arr[2]), gamma=float(arr[3]))

    @classmethod
    def default(cls) -> "RewardWeights":
        """CASR paper default: θ=0.8, others are our extensions."""
        return cls(theta=0.8, alpha=0.1, beta=0.3, gamma=0.2)

    @classmethod
    def random(cls, rng: Optional[np.random.Generator] = None) -> "RewardWeights":
        """Sample random weights for meta-model training data generation."""
        if rng is None:
            rng = np.random.default_rng()
        return cls(
            theta=float(rng.uniform(0.2, 0.95)),
            alpha=float(rng.uniform(0.0, 0.4)),
            beta=float(rng.uniform(0.1, 0.6)),
            gamma=float(rng.uniform(0.05, 0.4)),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Reward calculator
# ──────────────────────────────────────────────────────────────────────────────

class RewardCalculator:
    """
    Computes the extended CASR+ reward given per-step metrics.

    Usage:
        calc = RewardCalculator(sla_threshold=0.15)
        reward, components = calc.compute(metrics, prev_metrics, weights, prev_caps)
    """

    def __init__(
        self,
        sla_threshold: float = 0.15,   # 15% cold start rate = SLA limit
        sla_slope: float = 20.0,        # Steepness of sigmoid near threshold
        sla_scale: float = 1.0,         # Maximum SLA penalty magnitude
    ):
        self.sla_threshold = sla_threshold
        self.sla_slope = sla_slope
        self.sla_scale = sla_scale

        # Running normalizers for each component
        self._norm_r1 = _RunningNormalizer()
        self._norm_r2 = _RunningNormalizer()
        self._norm_r3 = _RunningNormalizer()
        self._norm_r4 = _RunningNormalizer()

    def compute(
        self,
        metrics: dict,
        prev_metrics: Optional[dict],
        weights: RewardWeights,
        prev_capacities: Optional[list] = None,
        current_capacities: Optional[list] = None,
    ) -> tuple:
        """
        Compute the reward for one RL step.

        Parameters
        ----------
        metrics        : Current step metrics from SCache.get_step_metrics()
        prev_metrics   : Previous step metrics (for R2 delta WMT)
        weights        : RewardWeights (θ, α, β, γ)
        prev_caps      : Queue capacities before action
        current_caps   : Queue capacities after action

        Returns
        -------
        (reward: float, components: dict)
            components contains individual R1..R4 and SLA values for logging.
        """
        total_invoc = max(1, metrics["total_invocations"])

        # ── R1: Cold start count (normalised by invocations for scale invariance) ─
        r1_raw = metrics["total_cold_starts"]

        # ── R2: Change in Wasted Memory Time (ΔWMT) ────────────────────────────
        prev_wmt = (prev_metrics["total_wmt"] if prev_metrics else 0.0)
        r2_raw = max(0.0, metrics["total_wmt"] - prev_wmt)

        # ── R3: Oscillation – how much capacities changed ───────────────────────
        if prev_capacities is not None and current_capacities is not None:
            cap_change = sum(
                abs(c_new - c_old)
                for c_new, c_old in zip(current_capacities, prev_capacities)
            )
        else:
            cap_change = 0.0
        r3_raw = float(cap_change)

        # ── R4: Cold-Start Overhead (CSO) – CASR's core additional metric ───────
        # CSO = Σ (overhead_ms × cold_starts) for each queue
        r4_raw = metrics["total_cso"]

        # ── Smooth SLA penalty ───────────────────────────────────────────────────
        cold_start_rate = metrics["total_cold_starts"] / total_invoc
        # Sigmoid centred at sla_threshold; fires smoothly as rate approaches/exceeds threshold
        sla_raw = self.sla_scale / (
            1.0 + np.exp(-self.sla_slope * (cold_start_rate - self.sla_threshold))
        )

        # ── Update normalizers ───────────────────────────────────────────────────
        self._norm_r1.update(r1_raw)
        self._norm_r2.update(r2_raw)
        self._norm_r3.update(r3_raw)
        self._norm_r4.update(r4_raw)

        # ── Normalize ────────────────────────────────────────────────────────────
        r1 = self._norm_r1.normalize(r1_raw)
        r2 = self._norm_r2.normalize(r2_raw)
        r3 = self._norm_r3.normalize(r3_raw)
        r4 = self._norm_r4.normalize(r4_raw)
        # SLA is already in [0, sla_scale] range; no normalisation needed

        θ = weights.theta
        α = weights.alpha
        β = weights.beta
        γ = weights.gamma

        # ── Final reward (negative = we want to minimise all terms) ─────────────
        # Base CASR term: -(θ·R1_norm + (1-θ)·R2_norm)
        base = θ * r1 + (1.0 - θ) * r2
        # Extension terms
        extension = α * r3 + β * sla_raw + γ * r4

        reward = -(base + extension)

        components = {
            "r1_raw": r1_raw,
            "r2_raw": r2_raw,
            "r3_raw": r3_raw,
            "r4_raw": r4_raw,
            "sla_raw": sla_raw,
            "r1_norm": r1,
            "r2_norm": r2,
            "r3_norm": r3,
            "r4_norm": r4,
            "cold_start_rate": cold_start_rate,
            "reward_base": -base,
            "reward_extension": -extension,
            "reward_total": reward,
        }

        return reward, components

    def reset(self):
        """Reset normalizers at the start of a new episode."""
        self._norm_r1 = _RunningNormalizer()
        self._norm_r2 = _RunningNormalizer()
        self._norm_r3 = _RunningNormalizer()
        self._norm_r4 = _RunningNormalizer()
