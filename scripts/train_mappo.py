#!/usr/bin/env python3
"""Train MAPPO on SkyArena2D."""
import argparse
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from skyarena2d.rl.utils.config import load_mappo_config
from skyarena2d.rl.algo.mappo_trainer import SkyArenaMAPPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mappo_skyarena.yaml")
    parser.add_argument("--total_env_steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_mappo_config(args.config)
    if args.total_env_steps:
        cfg["train"]["total_env_steps"] = args.total_env_steps
    if args.device:
        cfg["train"]["device"] = args.device

    trainer = SkyArenaMAPPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
