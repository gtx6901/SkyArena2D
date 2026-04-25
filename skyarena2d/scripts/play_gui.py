from __future__ import annotations

import argparse
import sys
import time
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
    args = parser.parse_args()

    if args.red not in RULES or args.blue not in RULES:
        raise ValueError("Unknown rule name")

    config = load_config(args.config)
    env = SkyArenaEngine(config, render_mode="human")
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
