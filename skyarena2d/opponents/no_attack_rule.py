from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseRuleOpponent


class NoAttackRuleOpponent(BaseRuleOpponent):
    def act(self, side_obs: dict[str, Any], side: str, step_count: int) -> dict[str, np.ndarray]:
        _ = step_count
        default_course = 0.0 if side == "red" else 180.0
        fighter_action, detector_action = self._base_actions(
            side_obs,
            default_course,
            radar_freq=1,
            jammer_freq=0,
        )

        for i, f in enumerate(side_obs["raw"]["fighter_obs_list"]):
            fighter_action[i, 0] = float(f.get("course", default_course))
            fighter_action[i, 3] = 0
        for i, d in enumerate(side_obs["raw"]["detector_obs_list"]):
            detector_action[i, 0] = float(d.get("course", default_course))
        return {"fighter_action": fighter_action, "detector_action": detector_action}
