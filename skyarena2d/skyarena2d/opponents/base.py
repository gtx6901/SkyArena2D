from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RuleConfig:
    radar_freq: int = 1
    jammer_freq: int = 0


class BaseRuleOpponent:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, side_obs: dict[str, Any], side: str, step_count: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _base_actions(
        self,
        side_obs: dict[str, Any],
        course_value: float,
        radar_freq: int = 1,
        jammer_freq: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        fighters = side_obs["raw"]["fighter_obs_list"]
        detectors = side_obs["raw"]["detector_obs_list"]
        fighter_action = np.zeros((len(fighters), 4), dtype=np.float32)
        detector_action = np.zeros((len(detectors), 2), dtype=np.float32)
        if len(fighters):
            fighter_action[:, 0] = course_value
            fighter_action[:, 1] = radar_freq
            fighter_action[:, 2] = jammer_freq
            fighter_action[:, 3] = 0
        if len(detectors):
            detector_action[:, 0] = course_value
            detector_action[:, 1] = radar_freq
        return fighter_action, detector_action

    def _choose_visible_target(self, fighter_obs: dict[str, Any]) -> tuple[int, float]:
        visible = fighter_obs.get("r_visible_list", [])
        if not visible:
            return 0, 0.0
        nearest = min(visible, key=lambda x: float(x.get("distance", 1e9)))
        return int(nearest["id"]), float(nearest.get("distance", 0.0))
