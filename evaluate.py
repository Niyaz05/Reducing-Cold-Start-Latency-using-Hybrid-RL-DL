"""
evaluate.py
-----------
Evaluates a trained CASR+ model and compares it against three baselines:

    1. Fixed       – Containers kept alive for fixed 10 minutes (platform default)
    2. LCS / LRU   – Least Recently Used eviction, no overhead-awareness
    3. CASR base   – CASR with fixed θ=0.8, no extended reward terms
    4. CASR+       – Our model (PPO + extended reward + meta-model weights)

Metrics (matching CASR paper Section 5.1.5):
    Cold Start Rate   = cold_starts / total_invocations
    Avg CSO           = total_cold_start_overhead_ms / total_invocations
    Avg WMT           = total_wasted_memory_time_ms / total_invocations

USAGE:
    python evaluate.py --data_dir ../data --model_path ../checkpoints/ppo_latest.pt
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import load_azure_dataset, compute_workload_features
from environment import CASREnv
from ppo_agent import PPOAgent
from meta_model import MetaModel
from reward import RewardWeights


# ──────────────────────────────────────────────────────────────────────────────
# Baseline: Fixed keep-alive (10 minutes)
# ──────────────────────────────────────────────────────────────────────────────

class FixedKeepAliveBaseline:
    """
    Containers are kept alive for a fixed number of invocations.
    Simple dict-based LRU with fixed capacity.
    """
    CAPACITY = 350  # Fixed total capacity across all functions

    def __init__(self):
        self._cache = OrderedDict()
        self._stats = {"invocations": 0, "cold_starts": 0, "cso_ms": 0.0, "wmt_ms": 0.0}

    def process(self, func_id: str, cold_start_ms: float, memory_mb: float):
        self._stats["invocations"] += 1
        if func_id in self._cache:
            self._cache.move_to_end(func_id)
        else:
            self._stats["cold_starts"] += 1
            self._stats["cso_ms"] += cold_start_ms
            self._cache[func_id] = memory_mb
            while len(self._cache) > self.CAPACITY:
                self._cache.popitem(last=False)
        # WMT: all cached containers accumulate idle time
        self._stats["wmt_ms"] += len(self._cache) * 1.0

    def get_metrics(self) -> dict:
        n = max(1, self._stats["invocations"])
        return {
            "cold_start_rate": self._stats["cold_starts"] / n,
            "avg_cso_ms": self._stats["cso_ms"] / n,
            "avg_wmt_ms": self._stats["wmt_ms"] / n,
            "name": "Fixed",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Baseline: LCS / LRU
# ──────────────────────────────────────────────────────────────────────────────

class LRUBaseline:
    """LRU eviction, overhead-unaware, no scaling."""
    CAPACITY = 350

    def __init__(self):
        self._cache = OrderedDict()
        self._stats = {"invocations": 0, "cold_starts": 0, "cso_ms": 0.0, "wmt_ms": 0.0}

    def process(self, func_id: str, cold_start_ms: float, memory_mb: float):
        self._stats["invocations"] += 1
        if func_id in self._cache:
            self._cache.move_to_end(func_id)
        else:
            self._stats["cold_starts"] += 1
            self._stats["cso_ms"] += cold_start_ms
            if len(self._cache) >= self.CAPACITY:
                self._cache.popitem(last=False)
            self._cache[func_id] = memory_mb
        self._stats["wmt_ms"] += len(self._cache) * 1.0

    def get_metrics(self) -> dict:
        n = max(1, self._stats["invocations"])
        return {
            "cold_start_rate": self._stats["cold_starts"] / n,
            "avg_cso_ms": self._stats["cso_ms"] / n,
            "avg_wmt_ms": self._stats["wmt_ms"] / n,
            "name": "LRU (LCS)",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────────────────────

def _run_baseline(baseline, data: pd.DataFrame) -> dict:
    """Run a simple baseline (no RL) through the dataset."""
    for row in data.itertuples(index=False):
        for _ in range(min(int(row.count), 50)):
            baseline.process(str(row.func_id), float(row.cold_start_ms), float(row.memory_mb))
    return baseline.get_metrics()


def _run_casr_env(
    data: pd.DataFrame,
    agent: PPOAgent,
    weights: RewardWeights,
    step_size: int = 500,
    label: str = "CASR+",
) -> dict:
    """Run a PPO agent through the environment and collect metrics."""
    env = CASREnv(data=data, weights=weights, step_size=step_size)
    state, _ = env.reset(options={"shuffle": False})

    total_cold = 0
    total_invoc = 0
    total_cso = 0.0
    total_wmt = 0.0
    done = False

    while not done:
        action, _, _ = agent.select_action(state)
        state, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_cold += info.get("r1_raw", 0)
        total_invoc += env._scache.queues[0].step_invocations + \
                       env._scache.queues[1].step_invocations + \
                       env._scache.queues[2].step_invocations
        total_cso += info.get("r4_raw", 0)
        total_wmt += info.get("r2_raw", 0)

    n = max(1, total_invoc)
    return {
        "cold_start_rate": total_cold / n,
        "avg_cso_ms": total_cso / n,
        "avg_wmt_ms": total_wmt / n,
        "name": label,
    }


def evaluate(
    data_dir: str,
    model_path: str,
    meta_model_path: Optional[str] = None,
    sample_rows: int = 100_000,
    results_dir: str = "../results",
    seed: int = 42,
):
    """Full evaluation run."""
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("  CASR+ Evaluation")
    print("=" * 60)

    # Load data
    data = load_azure_dataset(data_dir, sample_rows=sample_rows, random_seed=seed)
    features = compute_workload_features(data)

    # Load PPO agent
    agent = PPOAgent()
    agent.load(model_path)
    agent.actor.eval()
    agent.critic.eval()

    # Load meta-model (optional)
    meta_model = MetaModel()
    if meta_model_path and os.path.exists(meta_model_path):
        meta_model.load(meta_model_path)

    results = []

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\nRunning baselines...")
    results.append(_run_baseline(FixedKeepAliveBaseline(), data))
    results.append(_run_baseline(LRUBaseline(), data))

    # ── CASR base (fixed θ=0.8, no extension) ─────────────────────────────────
    casr_base_weights = RewardWeights(theta=0.8, alpha=0.0, beta=0.0, gamma=0.0)
    results.append(_run_casr_env(data, agent, casr_base_weights, label="CASR base"))

    # ── CASR+ (meta-model weights) ────────────────────────────────────────────
    if meta_model.is_trained:
        casr_plus_weights = meta_model.predict(features)
    else:
        casr_plus_weights = RewardWeights.default()
    results.append(_run_casr_env(data, agent, casr_plus_weights, label="CASR+ (ours)"))

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'Method':<18} | {'CS Rate':>8} | {'Avg CSO (ms)':>12} | {'Avg WMT (ms)':>12}")
    print("-" * 58)
    for r in results:
        print(
            f"{r['name']:<18} | {r['cold_start_rate']:>8.4f} | "
            f"{r['avg_cso_ms']:>12.1f} | {r['avg_wmt_ms']:>12.1f}"
        )

    # ── Save results ───────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out_path = os.path.join(results_dir, "evaluation_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # ── Improvement vs Fixed baseline ─────────────────────────────────────────
    fixed_cs = results[0]["cold_start_rate"]
    ours_cs = results[-1]["cold_start_rate"]
    if fixed_cs > 0:
        improvement = (fixed_cs - ours_cs) / fixed_cs * 100
        print(f"\nCARS+ vs Fixed: {improvement:+.1f}% cold start rate improvement")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

from typing import Optional

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CASR+ model")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--meta_model_path", type=str, default=None)
    p.add_argument("--sample_rows", type=int, default=100_000)
    p.add_argument("--results_dir", type=str, default="../results")
    p.add_argument("--seed", type=int, default=42)
    return vars(p.parse_args())


if __name__ == "__main__":
    args = parse_args()
    evaluate(**args)
