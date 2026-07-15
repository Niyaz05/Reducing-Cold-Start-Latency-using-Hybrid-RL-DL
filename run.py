"""
run.py
------
Quick-start entry point.  Run this from the project root.

    python run.py --mode train --data_dir ./data
    python run.py --mode eval  --data_dir ./data

Put all your Azure Functions 2019 Excel/CSV files inside the ./data folder.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--data_dir", type=str, default="./data",
                   help="Folder containing Azure Functions dataset files")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--sample_rows", type=int, default=300_000)
    p.add_argument("--step_size", type=int, default=500)
    args = p.parse_args()

    if args.mode == "train":
        from train import train, DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG.copy()
        cfg["data_dir"] = args.data_dir
        cfg["episodes"] = args.episodes
        cfg["sample_rows"] = args.sample_rows
        cfg["step_size"] = args.step_size
        train(cfg)

    elif args.mode == "eval":
        from evaluate import evaluate
        evaluate(
            data_dir=args.data_dir,
            model_path="./checkpoints/ppo_latest.pt",
            meta_model_path="./checkpoints/meta_model.pkl",
            sample_rows=100_000,
        )

if __name__ == "__main__":
    main()
