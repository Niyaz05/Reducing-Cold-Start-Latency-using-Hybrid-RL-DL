"""
train.py
--------
Main training pipeline for CASR+.

Training loop (Algorithm 2 from CASR paper, extended):
  For each episode:
    1. Extract workload features from dataset batch.
    2. If meta-model trained: predict weights (θ,α,β,γ).
       If not trained: use random weights (exploration phase).
    3. Initialise CASREnv with current weights.
    4. Run episode: agent ↔ environment interaction.
       Every UPDATE_INTERVAL steps: run PPO update.
    5. Record episode outcome.
    6. Every META_TRAIN_INTERVAL episodes: retrain meta-model.
    7. Save checkpoint.

USAGE:
    python train.py --data_dir ../data --episodes 200

    # Or with custom settings:
    python train.py --data_dir ../data --episodes 200 \\
        --sample_rows 300000 --step_size 500 --update_interval 200
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path when running directly
sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import load_azure_dataset, compute_workload_features
from environment import CASREnv
from ppo_agent import PPOAgent
from meta_model import MetaModel
from reward import RewardWeights


# ──────────────────────────────────────────────────────────────────────────────
# Training configuration (all overridable via CLI)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "data_dir": "../data",
    "checkpoints_dir": "../checkpoints",
    "results_dir": "../results",
    "episodes": 200,
    "sample_rows": 300_000,
    "step_size": 500,
    "update_interval": 200,     # PPO update every N steps
    "meta_train_interval": 10,  # Retrain meta-model every N episodes
    "sla_threshold": 0.15,
    "seed": 42,
    "save_every": 20,           # Save checkpoint every N episodes
}


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

class TrainingLogger:
    def __init__(self, results_dir: str):
        os.makedirs(results_dir, exist_ok=True)
        self._log_path = os.path.join(results_dir, "training_log.jsonl")
        self._episode_log = []

    def log_episode(self, episode: int, metrics: dict):
        record = {"episode": episode, **metrics}
        self._episode_log.append(record)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def print_summary(self, episode: int, metrics: dict):
        print(
            f"Ep {episode:4d} | "
            f"Reward={metrics.get('total_reward', 0):+7.3f} | "
            f"CS_rate={metrics.get('cold_start_rate', 0):.3f} | "
            f"θ={metrics.get('theta', 0):.2f} "
            f"α={metrics.get('alpha', 0):.2f} "
            f"β={metrics.get('beta', 0):.2f} "
            f"γ={metrics.get('gamma', 0):.2f} | "
            f"MetaTrained={metrics.get('meta_trained', False)}"
        )

    def save_final(self):
        df = pd.DataFrame(self._episode_log)
        df.to_csv(
            os.path.join(os.path.dirname(self._log_path), "training_results.csv"),
            index=False
        )
        print(f"[Logger] Saved training results CSV.")


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(config: dict):
    os.makedirs(config["checkpoints_dir"], exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)

    rng = np.random.default_rng(config["seed"])
    logger = TrainingLogger(config["results_dir"])

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("  CASR+ Training Pipeline")
    print("=" * 60)
    print(f"Loading dataset from: {config['data_dir']}")

    data = load_azure_dataset(
        data_dir=config["data_dir"],
        sample_rows=config["sample_rows"],
        random_seed=config["seed"],
    )

    # ── Initialise components ─────────────────────────────────────────────────
    meta_model = MetaModel(n_warmup=20)

    # Load existing meta-model if available
    meta_path = os.path.join(config["checkpoints_dir"], "meta_model.pkl")
    if os.path.exists(meta_path):
        meta_model.load(meta_path)

    agent = PPOAgent(
        state_dim=21,
        action_dim=27,
        gamma=0.63,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        epochs=10,
        mini_batch_size=20,
    )

    # Load existing PPO checkpoint if available
    ppo_path = os.path.join(config["checkpoints_dir"], "ppo_latest.pt")
    if os.path.exists(ppo_path):
        agent.load(ppo_path)

    # ── Training episodes ─────────────────────────────────────────────────────
    print(f"\nStarting training: {config['episodes']} episodes")
    print(f"Step size: {config['step_size']} invocations/step")
    print(f"SLA threshold: {config['sla_threshold']*100:.0f}%\n")

    for episode in range(1, config["episodes"] + 1):

        # ── Sample a batch of data for this episode ───────────────────────────
        n_episode = min(config["sample_rows"], len(data))
        episode_data = data.sample(n=n_episode, random_state=int(rng.integers(0, 10000)))
        episode_data = episode_data.reset_index(drop=True)

        # ── Extract workload features for meta-model ──────────────────────────
        features = compute_workload_features(episode_data)

        # ── Get weights for this episode ──────────────────────────────────────
        if meta_model.is_trained:
            weights = meta_model.predict(features)
        else:
            # Warmup: random weights for exploration
            weights = RewardWeights.random(rng)

        # ── Initialise environment ─────────────────────────────────────────────
        env = CASREnv(
            data=episode_data,
            weights=weights,
            step_size=config["step_size"],
            sla_threshold=config["sla_threshold"],
        )

        state, _ = env.reset(options={"shuffle": True})
        done = False
        step_count = 0

        # ── Episode loop ───────────────────────────────────────────────────────
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(
                state=state,
                action=action,
                log_prob=log_prob,
                reward=reward,
                next_state=next_state,
                done=done,
                value=value,
            )

            state = next_state
            step_count += 1

            # PPO update trigger
            if agent.buffer.count >= config["update_interval"] or done:
                agent.update()

        # ── Episode ended ──────────────────────────────────────────────────────
        summary = env.episode_summary()

        # ── Record for meta-model ──────────────────────────────────────────────
        meta_model.record_episode(
            features=features,
            weights=weights,
            cold_start_rate=summary["cold_start_rate"],
            total_wmt=sum(q.step_wmt for q in env._scache.queues),
        )

        # ── Retrain meta-model periodically ───────────────────────────────────
        if episode % config["meta_train_interval"] == 0:
            meta_model.train()
            meta_model.save(meta_path)

        # ── Logging ───────────────────────────────────────────────────────────
        episode_metrics = {
            **summary,
            "theta": weights.theta,
            "alpha": weights.alpha,
            "beta": weights.beta,
            "gamma": weights.gamma,
            "meta_trained": meta_model.is_trained,
            "meta_n_records": meta_model.n_episodes_collected,
        }
        logger.log_episode(episode, episode_metrics)
        logger.print_summary(episode, episode_metrics)

        # ── Save checkpoint ───────────────────────────────────────────────────
        if episode % config["save_every"] == 0:
            ckpt_path = os.path.join(
                config["checkpoints_dir"], f"ppo_ep{episode:04d}.pt"
            )
            agent.save(ckpt_path)
            agent.save(ppo_path)  # Also update "latest"

    # ── Training complete ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    agent.save(ppo_path)
    meta_model.save(meta_path)
    logger.save_final()

    # Final summary stats
    all_logs = pd.read_csv(os.path.join(config["results_dir"], "training_results.csv"))
    last_20 = all_logs.tail(20)
    print(f"\nFinal 20 episodes average:")
    print(f"  Cold start rate : {last_20['cold_start_rate'].mean():.3f}")
    print(f"  Total reward    : {last_20['total_reward'].mean():.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CASR+ PPO agent")
    p.add_argument("--data_dir", type=str, default=DEFAULT_CONFIG["data_dir"])
    p.add_argument("--checkpoints_dir", type=str, default=DEFAULT_CONFIG["checkpoints_dir"])
    p.add_argument("--results_dir", type=str, default=DEFAULT_CONFIG["results_dir"])
    p.add_argument("--episodes", type=int, default=DEFAULT_CONFIG["episodes"])
    p.add_argument("--sample_rows", type=int, default=DEFAULT_CONFIG["sample_rows"])
    p.add_argument("--step_size", type=int, default=DEFAULT_CONFIG["step_size"])
    p.add_argument("--update_interval", type=int, default=DEFAULT_CONFIG["update_interval"])
    p.add_argument("--meta_train_interval", type=int, default=DEFAULT_CONFIG["meta_train_interval"])
    p.add_argument("--sla_threshold", type=float, default=DEFAULT_CONFIG["sla_threshold"])
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument("--save_every", type=int, default=DEFAULT_CONFIG["save_every"])
    return vars(p.parse_args())


if __name__ == "__main__":
    config = parse_args()
    train(config)
