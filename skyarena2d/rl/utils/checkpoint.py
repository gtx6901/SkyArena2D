"""Checkpoint utilities for SkyArena MAPPO training."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def build_experiment_name(train_cfg: Dict[str, Any]) -> str:
    explicit = str(train_cfg.get("experiment_name", "")).strip()
    if explicit:
        return explicit
    return time.strftime("skyarena_mappo_%Y%m%d_%H%M%S")


def experiment_dir(train_cfg: Dict[str, Any]) -> Path:
    root = Path(str(train_cfg.get("train_dir", "train_dir/skyarena_mappo")))
    return root / build_experiment_name(train_cfg)


def checkpoint_dir(train_cfg: Dict[str, Any]) -> Path:
    return experiment_dir(train_cfg) / "checkpoints"


def tensorboard_dir(train_cfg: Dict[str, Any]) -> Path:
    return experiment_dir(train_cfg) / "tb"


def ensure_run_dirs(train_cfg: Dict[str, Any]) -> Dict[str, Path]:
    exp_dir = experiment_dir(train_cfg)
    ckpt_dir = checkpoint_dir(train_cfg)
    tb_dir = tensorboard_dir(train_cfg)
    for path in (exp_dir, ckpt_dir, tb_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"exp_dir": exp_dir, "ckpt_dir": ckpt_dir, "tb_dir": tb_dir}


def save_run_config(run_dirs: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    path = run_dirs["exp_dir"] / "config.resolved.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def latest_checkpoint(train_cfg: Dict[str, Any]) -> Optional[Path]:
    ckpt_dir = checkpoint_dir(train_cfg)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(
    *,
    train_cfg: Dict[str, Any],
    actor,
    critic,
    optimizer,
    env_steps: int,
    update_idx: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    ckpt_dir = checkpoint_dir(train_cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{env_steps:09d}.pt"
    payload = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "env_steps": int(env_steps),
        "update_idx": int(update_idx),
        "saved_at_unix": float(time.time()),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, actor, critic, optimizer=None, map_location="cpu") -> Dict[str, Any]:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
