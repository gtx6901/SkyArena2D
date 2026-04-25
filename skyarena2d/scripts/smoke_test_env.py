from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyarena2d.core.config import load_config
from skyarena2d.core.engine import SkyArenaEngine


def random_maca_actions(config, rng: np.random.Generator) -> dict[str, dict[str, np.ndarray]]:
    red_enemy = config.teams.blue_fighters + config.teams.blue_detectors
    blue_enemy = config.teams.red_fighters + config.teams.red_detectors

    def side_actions(n_f: int, n_d: int, max_enemy: int) -> dict[str, np.ndarray]:
        fighter = np.zeros((n_f, 4), dtype=np.float32)
        detector = np.zeros((n_d, 2), dtype=np.float32)
        if n_f > 0:
            fighter[:, 0] = rng.integers(0, 360, size=n_f)
            fighter[:, 1] = rng.integers(0, config.radar.freq_count + 1, size=n_f)
            fighter[:, 2] = rng.integers(0, config.radar.freq_count + 2, size=n_f)
            fire_mode = rng.integers(0, 3, size=n_f)
            target = rng.integers(1, max_enemy + 1, size=n_f) if max_enemy > 0 else np.zeros((n_f,))
            fighter[:, 3] = np.where(
                fire_mode == 0,
                0,
                np.where(fire_mode == 1, target, max_enemy + target),
            )
        if n_d > 0:
            detector[:, 0] = rng.integers(0, 360, size=n_d)
            detector[:, 1] = rng.integers(0, config.radar.freq_count + 1, size=n_d)
        return {"fighter_action": fighter, "detector_action": detector}

    return {
        "red": side_actions(config.teams.red_fighters, config.teams.red_detectors, red_enemy),
        "blue": side_actions(config.teams.blue_fighters, config.teams.blue_detectors, blue_enemy),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_10v10_full.yaml")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    env = SkyArenaEngine(config)
    obs, info = env.reset(seed=7)
    rng = np.random.default_rng(7)

    print("Initial red modern shape:", obs["red"]["modern"]["self"].shape)
    print("Initial blue modern shape:", obs["blue"]["modern"]["self"].shape)
    print("Initial metrics:", info["metrics"])

    for step in range(args.steps):
        action = random_maca_actions(config, rng)
        obs, reward, done, truncated, info = env.step(action)

        for side in ("red", "blue"):
            for key in ("self", "allies", "enemies"):
                arr = obs[side]["modern"][key]
                if np.isnan(arr).any():
                    raise AssertionError(f"NaN detected in {side} modern {key}")

        print(
            f"step={step + 1} red_reward={reward['red']:.3f} blue_reward={reward['blue']:.3f} done={done} truncated={truncated}"
        )
        if done:
            break

    print("Final metrics:", info["metrics"])
    env.close()


if __name__ == "__main__":
    main()
