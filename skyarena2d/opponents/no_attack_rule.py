from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseRuleOpponent


class NoAttackRuleOpponent(BaseRuleOpponent):
    """Flies straight with boundary bounce. Never fires, never chases."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        map_width: float = 3000.0,
        map_height: float = 4000.0,
        margin: float = 80.0,
    ):
        super().__init__(seed=seed)
        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self.margin = float(margin)

    def _reflect_course(self, pos_x: float, pos_y: float, course: float) -> float:
        """Reflect course angle (degrees) if near boundary. 2D billiard-style bounce."""
        heading = float(course) % 360.0
        if pos_x <= self.margin or pos_x >= self.map_width - self.margin:
            heading = (180.0 - heading) % 360.0
        if pos_y <= self.margin or pos_y >= self.map_height - self.margin:
            heading = (-heading) % 360.0
        return heading

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
            course = float(f.get("course", default_course))
            pos_x = float(f.get("pos_x", 0.0))
            pos_y = float(f.get("pos_y", 0.0))
            fighter_action[i, 0] = self._reflect_course(pos_x, pos_y, course)
            fighter_action[i, 3] = 0
        for i, d in enumerate(side_obs["raw"]["detector_obs_list"]):
            course = float(d.get("course", default_course))
            pos_x = float(d.get("pos_x", 0.0))
            pos_y = float(d.get("pos_y", 0.0))
            detector_action[i, 0] = self._reflect_course(pos_x, pos_y, course)
        return {"fighter_action": fighter_action, "detector_action": detector_action}
