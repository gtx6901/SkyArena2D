from __future__ import annotations

import argparse
import sys
import time
import numpy as np
from pathlib import Path

try:
    import pygame
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pygame-ce is required for play_gui.py") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyarena2d.core.config import load_config
from skyarena2d.core.engine import SkyArenaEngine
from skyarena2d.opponents import RULES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--red", default="rush_rule")
    parser.add_argument("--blue", default="patrol_rule")
    parser.add_argument("--config", default="configs/env_10v10_full.yaml")
    parser.add_argument("--speed", type=float, default=30.0, help="steps per second")
    parser.add_argument("--debug", action="store_true", help="Print debug info each step")
    parser.add_argument("--no-jamming", action="store_true", help="Disable jamming in config")
    parser.add_argument("--allow-passive-fire", action="store_true", help="Enable passive fire in config")
    args = parser.parse_args()

    if args.red not in RULES or args.blue not in RULES:
        raise ValueError("Unknown rule name")

    config = load_config(args.config)
    if args.no_jamming:
        config.jamming.enabled = False
    if args.allow_passive_fire:
        config.weapon.allow_passive_fire = True

    env = SkyArenaEngine(config, render_mode="human")
    # Ensure pygame video system is initialized so pygame.event.* works
    try:
        pygame.init()
        pygame.display.init()
    except Exception:  # pragma: no cover - ignore environments without a display
        pass
    red_rule = RULES[args.red]()
    blue_rule = RULES[args.blue]()

    obs, _ = env.reset(seed=0)
    paused = False
    single_step = False
    running = True

    print("Controls: SPACE pause/resume, N single-step, D toggle debug overlay, ESC quit")
    while running:
        start = time.perf_counter()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_n:
                    single_step = True
                elif event.key == pygame.K_d:
                    if env._renderer is not None:
                        env._renderer.toggle_debug_overlay()

        if not paused or single_step:
            state = env.get_state()
            red_action = red_rule.act(obs["red"], "red", state.step_count)
            blue_action = blue_rule.act(obs["blue"], "blue", state.step_count)
            obs, _, done, _, info = env.step({"red": red_action, "blue": blue_action})
            single_step = False
            if args.debug:
                try:
                    state = env.get_state()
                    cache = state.cache
                    if cache is not None:
                        red_vis = int(np.sum(cache.red_visible))
                        blue_vis = int(np.sum(cache.blue_visible))
                        red_long = int(np.sum(cache.red_fireable_long))
                        red_short = int(np.sum(cache.red_fireable_short))
                        blue_long = int(np.sum(cache.blue_fireable_long))
                        blue_short = int(np.sum(cache.blue_fireable_short))
                        print(
                            f"Step {state.step_count}: red_vis={red_vis} blue_vis={blue_vis} "
                            f"red_long={red_long} red_short={red_short} blue_long={blue_long} blue_short={blue_short}"
                        )
                        if cache.launch_records:
                            for rec in cache.launch_records:
                                print(
                                    f"  LaunchRecord: {rec.attacker_side}#{rec.attacker_idx} -> {rec.target_side}#{rec.target_idx} "
                                    f"type={rec.missile_type} valid={rec.valid} reason={rec.reason} hit_prob={rec.hit_prob}"
                                )
                except Exception:
                    pass

            if done:
                print(
                    f"Episode done. winner={info['winner']} reason={info['reason']} red_kills={info['metrics']['red_kills']} blue_kills={info['metrics']['blue_kills']}"
                )
                obs, _ = env.reset()

        env.render("human")
        frame_budget = 1.0 / max(1.0, args.speed)
        elapsed = time.perf_counter() - start
        if elapsed < frame_budget:
            time.sleep(frame_budget - elapsed)

    env.close()


if __name__ == "__main__":
    main()
