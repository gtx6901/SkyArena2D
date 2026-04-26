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
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--gui_eval_interval", type=int, default=None)
    parser.add_argument("--disable_gui_eval", action="store_true")
    parser.add_argument("--gui_eval_human", action="store_true")
    parser.add_argument("--init_checkpoint", default="")
    args = parser.parse_args()

    cfg = load_mappo_config(args.config)
    if args.total_env_steps:
        cfg["train"]["total_env_steps"] = args.total_env_steps
    if args.device:
        cfg["train"]["device"] = args.device
    if args.save_interval:
        cfg["train"]["save_interval"] = args.save_interval
    if args.eval_interval:
        cfg["train"]["eval_interval"] = args.eval_interval
        cfg.setdefault("evaluation", {})["policy_eval_interval"] = args.eval_interval
    if args.gui_eval_interval:
        cfg.setdefault("evaluation", {})["gui_eval_interval"] = args.gui_eval_interval
    if args.disable_gui_eval:
        cfg.setdefault("evaluation", {})["gui_eval_enabled"] = False
    if args.gui_eval_human:
        cfg.setdefault("evaluation", {})["gui_eval_human"] = True
        cfg.setdefault("evaluation", {})["gui_eval_render_mode"] = "human"
    if args.init_checkpoint:
        cfg["train"]["init_checkpoint"] = args.init_checkpoint

    trainer = SkyArenaMAPPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
