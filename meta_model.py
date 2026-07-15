"""
meta_model.py
-------------
Meta-model: predicts the optimal reward weights (θ, α, β, γ) based on
workload characteristics extracted from the incoming data batch.

WHY A META-MODEL?
-----------------
CASR uses a fixed θ=0.8 across all workloads.  But different workloads
need different weight settings:

    High-overhead bursty workload → need high θ (punish cold starts hard)
    Steady low-traffic workload   → lower θ, prioritise memory efficiency
    Irregular function calls      → high β (enforce SLA tightly)

A trained meta-model observes workload features and predicts which weights
will minimise cold starts + memory waste for THAT specific workload.

TRAINING APPROACH (meta-learning loop):
1.  Run N_WARMUP episodes with random (θ, α, β, γ) combinations.
2.  After each episode, record:
        X = workload features (10 scalars)
        y = weights that produced the best outcome
3.  Once enough data exists, train an XGBoost model on (X → y).
4.  For subsequent episodes: predict weights from workload features.

This is essentially a simple form of automatic hyperparameter tuning,
where the hyperparameters are the reward function weights.

FEATURES (10 dimensions):
    mean_invocations_per_minute  – How busy is the workload?
    std_invocations_per_minute   – Is it bursty?
    mean_cold_start_ms           – How heavy are cold starts on average?
    std_cold_start_ms            – How heterogeneous are overhead values?
    frac_low_overhead            – Fraction of Q0 (0–1s) functions
    frac_medium_overhead         – Fraction of Q1 (1–60s) functions
    frac_high_overhead           – Fraction of Q2 (>60s) functions
    mean_memory_mb               – Average memory footprint
    n_unique_functions           – How diverse is the function set?
    total_invocations            – Total volume
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import os
import pickle

from reward import RewardWeights


# Minimum episodes before we start using the meta-model
N_WARMUP = 20

# Feature and target column names
FEATURE_COLS = [
    "mean_invocations_per_minute",
    "std_invocations_per_minute",
    "mean_cold_start_ms",
    "std_cold_start_ms",
    "frac_low_overhead",
    "frac_medium_overhead",
    "frac_high_overhead",
    "mean_memory_mb",
    "n_unique_functions",
    "total_invocations",
]
TARGET_COLS = ["theta", "alpha", "beta", "gamma"]


class MetaDataCollector:
    """
    Collects (workload_features, weights, outcome) tuples across episodes.
    The "best" weights for a workload are those that produced the lowest
    cold_start_rate weighted with memory_wmt.
    """

    def __init__(self, save_path: Optional[str] = None):
        self.save_path = save_path
        self._records: List[dict] = []

    def record(
        self,
        features: dict,
        weights: RewardWeights,
        cold_start_rate: float,
        total_wmt: float,
    ):
        """Record one episode's outcome."""
        # Composite score: minimise both cold starts and memory waste
        # Use the same objective as CASR: θ·CS + (1-θ)·WMT
        score = weights.theta * cold_start_rate + (1 - weights.theta) * (total_wmt / 1e8)
        record = {
            **features,
            "theta": weights.theta,
            "alpha": weights.alpha,
            "beta": weights.beta,
            "gamma": weights.gamma,
            "cold_start_rate": cold_start_rate,
            "total_wmt": total_wmt,
            "score": score,
        }
        self._records.append(record)

    def get_training_data(self, n_warmup: int = N_WARMUP) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Build training data: for each unique workload fingerprint,
        find the weights that produced the lowest score.

        Returns (X, y) DataFrames or (None, None) if not enough data.
        """
        if len(self._records) < n_warmup:
            return None, None

        df = pd.DataFrame(self._records)
        X = df[FEATURE_COLS]
        y = df[TARGET_COLS]
        return X, y

    def save(self):
        if self.save_path:
            with open(self.save_path, "wb") as f:
                pickle.dump(self._records, f)

    def load(self):
        if self.save_path and os.path.exists(self.save_path):
            with open(self.save_path, "rb") as f:
                self._records = pickle.load(f)

    @property
    def n_records(self) -> int:
        return len(self._records)


class MetaModel:
    """
    XGBoost-based meta-model that predicts (θ, α, β, γ) from workload features.

    Falls back to CASR's default weights (θ=0.8) during the warmup period.
    """

    def __init__(self, n_warmup: int = N_WARMUP):
        self.n_warmup = n_warmup
        self._model = None          # Will be an XGBoost MultiOutputRegressor
        self._scaler = None         # Feature scaler
        self._is_trained = False
        self._collector = MetaDataCollector()

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def n_episodes_collected(self) -> int:
        return self._collector.n_records

    def record_episode(
        self,
        features: dict,
        weights: RewardWeights,
        cold_start_rate: float,
        total_wmt: float,
    ):
        """Call after each episode to record outcome."""
        self._collector.record(features, weights, cold_start_rate, total_wmt)

    def train(self) -> bool:
        """
        Train the meta-model on collected data.
        Returns True if training succeeded.
        """
        X, y = self._collector.get_training_data(n_warmup=self.n_warmup)
        if X is None:
            print(f"[MetaModel] Not enough data yet "
                  f"({self._collector.n_records}/{self.n_warmup} episodes).")
            return False

        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.multioutput import MultiOutputRegressor
            from xgboost import XGBRegressor

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)

            base = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbosity=0,
            )
            self._model = MultiOutputRegressor(base)
            self._model.fit(X_scaled, y)
            self._is_trained = True
            print(f"[MetaModel] Trained on {len(X)} episodes.")
            return True

        except ImportError as e:
            print(f"[MetaModel] Could not train: {e}")
            print("[MetaModel] Install xgboost and scikit-learn to enable the meta-model.")
            return False

    def predict(self, features: dict) -> RewardWeights:
        """
        Predict optimal reward weights for the given workload features.

        Falls back to CASR default weights if model not trained.
        """
        if not self._is_trained:
            # During warmup: return random weights for exploration
            return RewardWeights.random()

        try:
            feat_arr = np.array(
                [features.get(col, 0.0) for col in FEATURE_COLS],
                dtype=np.float64,
            ).reshape(1, -1)

            scaled = self._scaler.transform(feat_arr)
            pred = self._model.predict(scaled)[0]

            # Clamp to valid ranges
            theta = float(np.clip(pred[0], 0.1, 0.95))
            alpha = float(np.clip(pred[1], 0.0, 0.5))
            beta = float(np.clip(pred[2], 0.0, 0.8))
            gamma = float(np.clip(pred[3], 0.0, 0.5))

            return RewardWeights(theta=theta, alpha=alpha, beta=beta, gamma=gamma)

        except Exception as e:
            print(f"[MetaModel] Prediction error: {e}. Using defaults.")
            return RewardWeights.default()

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "scaler": self._scaler,
                "is_trained": self._is_trained,
            }, f)
        print(f"[MetaModel] Saved → {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._scaler = data["scaler"]
        self._is_trained = data["is_trained"]
        print(f"[MetaModel] Loaded ← {path}")
