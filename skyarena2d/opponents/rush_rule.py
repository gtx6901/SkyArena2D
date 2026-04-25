from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseRuleOpponent


class RushRuleOpponent(BaseRuleOpponent):
    def __init__(self, seed: int | None = None, short_range_hint: float = 130.0) -> None:
        super().__init__(seed)
        self.short_range_hint = short_range_hint

    def act(self, side_obs: dict[str, Any], side: str, step_count: int) -> dict[str, np.ndarray]:
        _ = step_count
        base_course = 0.0 if side == "red" else 180.0
        fighter_action, detector_action = self._base_actions(
            side_obs,
            course_value=base_course,
            radar_freq=1,
            jammer_freq=1,
        )

        max_enemy = side_obs["modern"]["enemies"].shape[1]
        for i, fighter_obs in enumerate(side_obs["raw"]["fighter_obs_list"]):
            target_id, dist = self._choose_visible_target(fighter_obs)
            if target_id > 0:
                if dist <= self.short_range_hint:
                    fighter_action[i, 3] = max_enemy + target_id
                else:
                    fighter_action[i, 3] = target_id
            fighter_action[i, 0] = base_course

        for i in range(len(side_obs["raw"]["detector_obs_list"])):
            detector_action[i, 0] = base_course
            detector_action[i, 1] = 1
        return {"fighter_action": fighter_action, "detector_action": detector_action}
