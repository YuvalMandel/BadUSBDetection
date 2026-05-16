"""
Generate random GRU HP configs for SLURM array search.
Run once from repo root before submitting hpsearch_gru_array.sh.

Usage:
    python GRU/generate_gru_configs.py [--n-configs 128] [--seed 42]
"""
import argparse
import json
import os
import numpy as np


HIDDEN_DIMS  = [64, 128, 256]
BATCH_SIZES  = [16, 32, 64]
DROPOUT_VALS = np.round(np.arange(0.10, 0.45, 0.05), 2).tolist()
THRESH_VALS  = np.round(np.arange(0.30, 0.90, 0.05), 2).tolist()
LR_LOW, LR_HIGH = np.log(5e-5), np.log(2e-3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-configs", type=int, default=128)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out-dir",   default=os.path.join("results", "GRU", "hp_trials"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(args.n_configs):
        cfg = {
            "trial_idx":  i,
            "hidden_dim": int(rng.choice(HIDDEN_DIMS)),
            "num_layers": 2,
            "dropout":    float(rng.choice(DROPOUT_VALS)),
            "lr":         float(np.exp(rng.uniform(LR_LOW, LR_HIGH))),
            "batch_size": int(rng.choice(BATCH_SIZES)),
            "threshold":  float(rng.choice(THRESH_VALS)),
        }
        with open(os.path.join(args.out_dir, f"trial_{i:04d}.json"), "w") as fh:
            json.dump(cfg, fh, indent=2)

    print(f"Generated {args.n_configs} configs -> {args.out_dir}")


if __name__ == "__main__":
    main()
