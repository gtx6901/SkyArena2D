#!/usr/bin/env python3
"""Evaluate trained MAPPO policy on SkyArena2D."""
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
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--render_mode", choices=["human", "rgb_array"], default=None)
    parser.add_argument("--gui_eval_human", action="store_true")
    parser.add_argument("--kind", default="eval")
    args = parser.parse_args()

    cfg = load_mappo_config(args.config)
    if args.device:
        cfg["train"]["device"] = args.device

    trainer = SkyArenaMAPPOTrainer(cfg)
    render_mode = "human" if args.gui_eval_human else args.render_mode
    kind = args.kind
    if render_mode is not None and kind == "eval":
        kind = "gui_eval"
    trainer.evaluate(
        num_episodes=args.episodes,
        checkpoint_path=args.checkpoint,
        render_mode=render_mode,
        max_steps=args.max_steps,
        write_report=True,
        kind=kind,
    )


if __name__ == "__main__":
    main()
