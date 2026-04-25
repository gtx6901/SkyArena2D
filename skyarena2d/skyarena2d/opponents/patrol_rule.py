from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseRuleOpponent


class PatrolRuleOpponent(BaseRuleOpponent):
    def __init__(self, seed: int | None = None, sweep_period: int = 120) -> None:
        super().__init__(seed)
        self.sweep_period = max(20, sweep_period)

    def act(self, side_obs: dict[str, Any], side: str, step_count: int) -> dict[str, np.ndarray]:
        direction = 1.0 if (step_count // self.sweep_period) % 2 == 0 else -1.0
        base = 30.0 if side == "red" else 210.0
        course = (base + direction * 45.0) % 360.0
        fighter_action, detector_action = self._base_actions(
            side_obs,
            course_value=course,
            radar_freq=1,
            jammer_freq=0,
        )

        max_enemy = side_obs["modern"]["enemies"].shape[1]
        for i, fighter_obs in enumerate(side_obs["raw"]["fighter_obs_list"]):
            target_id, dist = self._choose_visible_target(fighter_obs)
            if target_id > 0:
                fighter_action[i, 3] = target_id if dist > 110.0 else max_enemy + target_id
                fighter_action[i, 0] = 0.0 if side == "red" else 180.0
        return {"fighter_action": fighter_action, "detector_action": detector_action}
