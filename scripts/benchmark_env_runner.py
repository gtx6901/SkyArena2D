#!/usr/bin/env python3
"""Measure raw SkyArena environment stepping throughput without PPO."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from skyarena2d.adapters.action_types import SkyArenaSideAction
from skyarena2d.rl.adapters.vector_env import build_env_runner
from skyarena2d.rl.utils.config import load_mappo_config


def _no_fire_action(context) -> SkyArenaSideAction:
    n_agents = context.num_fighters
    alive = np.asarray(context.alive, dtype=bool)
    radar = np.zeros(n_agents, dtype=np.int32)
    radar[alive] = 1
    return SkyArenaSideAction(
        course=np.asarray(context.current_heading, dtype=np.float32).copy(),
        radar_freq=radar,
        jammer_freq=np.zeros(n_agents, dtype=np.int32),
        fire_type=np.zeros(n_agents, dtype=np.int32),
        target_idx=np.full(n_agents, -1, dtype=np.int32),
    )


def _run_steps(runner, steps: int) -> int:
    completed = 0
    for _ in range(steps):
        actions = [_no_fire_action(context) for context in runner.action_contexts]
        results = runner.step_all(actions)
        done_indices = [index for index, result in enumerate(results) if result[2]]
        if done_indices:
            runner.reset_many(done_indices)
        completed += len(results)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/mappo_skyarena_baseline_v2_smoke.yaml"
    )
    parser.add_argument("--backend", choices=("serial", "subprocess"), required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()

    cfg = load_mappo_config(args.config)
    cfg["train"]["num_envs"] = args.num_envs
    cfg["train"]["env_backend"] = args.backend
    cfg["train"]["env_workers"] = args.workers
    runner = build_env_runner(cfg, args.num_envs)
    try:
        runner.reset_all()
        _run_steps(runner, max(args.warmup_steps, 0))
        start = time.perf_counter()
        env_steps = _run_steps(runner, max(args.steps, 1))
        elapsed = time.perf_counter() - start
    finally:
        runner.close()

    print(json.dumps({
        "backend": args.backend,
        "num_envs": args.num_envs,
        "workers": getattr(runner, "num_workers", 1),
        "env_steps": env_steps,
        "seconds": elapsed,
        "env_steps_per_second": env_steps / elapsed,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
