from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseRuleOpponent


class FixRuleLikeOpponent(BaseRuleOpponent):
    def __init__(
        self,
        seed: int | None = None,
        push_steps: int = 120,
        search_turn_amplitude: float = 60.0,
    ) -> None:
        super().__init__(seed)
        self.push_steps = push_steps
        self.search_turn_amplitude = search_turn_amplitude

    def act(self, side_obs: dict[str, Any], side: str, step_count: int) -> dict[str, np.ndarray]:
        if step_count < self.push_steps:
            base_course = 0.0 if side == "red" else 180.0
        else:
            swing = np.sin(step_count / 30.0) * self.search_turn_amplitude
            base_course = ((0.0 if side == "red" else 180.0) + swing) % 360.0

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
                fighter_action[i, 0] = (0.0 if side == "red" else 180.0)
                if dist <= 100.0:
                    fighter_action[i, 3] = max_enemy + target_id
                else:
                    fighter_action[i, 3] = target_id
            else:
                if step_count >= self.push_steps:
                    jitter = float(self.rng.uniform(-12.0, 12.0))
                    fighter_action[i, 0] = (fighter_action[i, 0] + jitter) % 360.0

        return {"fighter_action": fighter_action, "detector_action": detector_action}
