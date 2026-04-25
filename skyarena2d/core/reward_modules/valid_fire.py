from __future__ import annotations

import numpy as np

from .base import RewardModule


class ValidFireModule(RewardModule):
    """Reward for valid fire. Optional only_when_opportunity flag."""

    def __init__(self, coef: float = 0.02, only_when_opportunity: bool = False) -> None:
        self.coef = coef
        self.only_when_opportunity = only_when_opportunity

    @property
    def name(self) -> str:
        return "valid_fire"

    def compute(self, state, config, weapon_result, termination_result) -> tuple[np.ndarray, np.ndarray, dict]:
        red_delta = np.zeros((state.red.total_units,), dtype=np.float32)
        blue_delta = np.zeros((state.blue.total_units,), dtype=np.float32)

        if self.only_when_opportunity and weapon_result.red_fireable_long.size > 0:
            red_has_opp = np.any(weapon_result.red_fireable_long | weapon_result.red_fireable_short, axis=1)
            blue_has_opp = np.any(weapon_result.blue_fireable_long | weapon_result.blue_fireable_short, axis=1)
            red_delta[weapon_result.red_valid_fire & red_has_opp] += self.coef
            blue_delta[weapon_result.blue_valid_fire & blue_has_opp] += self.coef
        else:
            red_delta[weapon_result.red_valid_fire] += self.coef
            blue_delta[weapon_result.blue_valid_fire] += self.coef

        return red_delta, blue_delta, {
            "red": float(np.sum(red_delta)),
            "blue": float(np.sum(blue_delta)),
        }
