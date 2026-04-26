#!/usr/bin/env python3
"""Run the staged SkyArena2D MAPPO curriculum."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))


STAGES = [
    ("stage0_no_attack", "configs/mappo_skyarena_stage0_no_attack.yaml"),
    ("stage1_patrol", "configs/mappo_skyarena_stage1_patrol.yaml"),
    ("stage2_rush", "configs/mappo_skyarena_stage2_rush.yaml"),
    ("stage3_fix_like", "configs/mappo_skyarena_stage3_fix_like.yaml"),
    ("stage4_fix_v2", "configs/mappo_skyarena_stage4_fix_v2.yaml"),
]


def _stage_index(value: str) -> int:
    value = str(value).strip()
    if value.isdigit():
        idx = int(value)
        if 0 <= idx < len(STAGES):
            return idx
    for idx, (name, _) in enumerate(STAGES):
        if value == name:
            return idx
    valid = ", ".join([name for name, _ in STAGES])
    raise argparse.ArgumentTypeError(f"unknown stage {value!r}; use 0-4 or one of: {valid}")


def _latest_for_config(config_path: str, *, dry_run: bool = False) -> Path | None:
    try:
        from skyarena2d.rl.utils.checkpoint import latest_checkpoint
        from skyarena2d.rl.utils.config import load_mappo_config

        cfg = load_mappo_config(config_path)
        return latest_checkpoint(cfg["train"])
    except ModuleNotFoundError:
        if not dry_run:
            raise

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg["train"]
    root = Path(str(train_cfg.get("train_dir", "train_dir/skyarena_mappo")))
    experiment = str(train_cfg.get("experiment_name", "")).strip()
    ckpt_dir = root / experiment / "checkpoints"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def _format_command(cmd: list[str]) -> str:
    return " ".join(cmd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start_stage", type=_stage_index, default=0)
    parser.add_argument("--end_stage", type=_stage_index, default=len(STAGES) - 1)
    parser.add_argument("--disable_gui_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--total_env_steps_override", type=int, default=None)
    args = parser.parse_args()

    if args.start_stage > args.end_stage:
        parser.error("--start_stage must be <= --end_stage")

    previous_checkpoint: Path | None = None
    if args.start_stage > 0:
        previous_name, previous_config = STAGES[args.start_stage - 1]
        previous_checkpoint = _latest_for_config(previous_config, dry_run=args.dry_run)
        if previous_checkpoint is None and not args.dry_run:
            raise RuntimeError(f"No checkpoint found for previous stage {previous_name}: {previous_config}")

    for idx in range(args.start_stage, args.end_stage + 1):
        stage_name, config_path = STAGES[idx]
        init_checkpoint = None if idx == 0 else previous_checkpoint
        cmd = [
            sys.executable,
            "scripts/train_mappo.py",
            "--config",
            config_path,
            "--device",
            args.device,
        ]
        if init_checkpoint is not None:
            cmd.extend(["--init_checkpoint", str(init_checkpoint)])
        if args.disable_gui_eval:
            cmd.append("--disable_gui_eval")
        if args.total_env_steps_override is not None:
            cmd.extend(["--total_env_steps", str(args.total_env_steps_override)])

        if args.dry_run:
            print(f"[dry_run] {stage_name}: {_format_command(cmd)}", flush=True)
            if idx == 0:
                print("[dry_run] stage0_no_attack starts without init_checkpoint", flush=True)
            elif init_checkpoint is not None:
                print(f"[dry_run] {stage_name} would warm-start from {init_checkpoint}", flush=True)
            else:
                prev_name, _ = STAGES[idx - 1]
                print(
                    f"[dry_run] {stage_name} init_checkpoint will be resolved from {prev_name} after it completes",
                    flush=True,
                )
                latest_prev = _latest_for_config(STAGES[idx - 1][1], dry_run=True)
                if latest_prev is not None:
                    print(f"[dry_run] current latest for {prev_name}: {latest_prev}", flush=True)
            previous_checkpoint = _latest_for_config(config_path, dry_run=True)
            continue

        print(f"[curriculum] starting {stage_name}: {_format_command(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        previous_checkpoint = _latest_for_config(config_path)
        if previous_checkpoint is None and idx < args.end_stage:
            raise RuntimeError(f"No checkpoint found after {stage_name}: {config_path}")
        print(f"[curriculum] latest checkpoint for {stage_name}: {previous_checkpoint}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
