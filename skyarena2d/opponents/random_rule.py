from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseRuleOpponent


class RandomRuleOpponent(BaseRuleOpponent):
    def __init__(self, seed: int | None = None, freq_count: int = 10) -> None:
        super().__init__(seed)
        self.freq_count = freq_count

    def act(self, side_obs: dict[str, Any], side: str, step_count: int) -> dict[str, np.ndarray]:
        _ = side, step_count
        fighters = side_obs["raw"]["fighter_obs_list"]
        detectors = side_obs["raw"]["detector_obs_list"]
        max_enemy = side_obs["modern"]["enemies"].shape[1]

        fighter_action = np.zeros((len(fighters), 4), dtype=np.float32)
        detector_action = np.zeros((len(detectors), 2), dtype=np.float32)

        for i in range(len(fighters)):
            fighter_action[i, 0] = float(self.rng.integers(0, 360))
            fighter_action[i, 1] = float(self.rng.integers(0, self.freq_count + 1))
            fighter_action[i, 2] = float(self.rng.integers(0, self.freq_count + 2))
            fire_mode = int(self.rng.integers(0, 3))
            target = int(self.rng.integers(1, max_enemy + 1)) if max_enemy > 0 else 0
            if fire_mode == 0 or target == 0:
                fighter_action[i, 3] = 0
            elif fire_mode == 1:
                fighter_action[i, 3] = target
            else:
                fighter_action[i, 3] = max_enemy + target

        for i in range(len(detectors)):
            detector_action[i, 0] = float(self.rng.integers(0, 360))
            detector_action[i, 1] = float(self.rng.integers(0, self.freq_count + 1))

        return {"fighter_action": fighter_action, "detector_action": detector_action}
