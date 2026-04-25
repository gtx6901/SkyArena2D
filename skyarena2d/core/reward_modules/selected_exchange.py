from __future__ import annotations

import numpy as np

from .base import RewardModule


class SelectedExchangeModule(RewardModule):
    """Reward based on expected exchange ratio of selected (launched) targets."""

    def __init__(self, coef: float = 0.05) -> None:
        self.coef = coef

    @property
    def name(self) -> str:
        return "selected_exchange"

    def compute(self, state, config, weapon_result, termination_result) -> tuple[np.ndarray, np.ndarray, dict]:
        red_delta = np.zeros((state.red.total_units,), dtype=np.float32)
        blue_delta = np.zeros((state.blue.total_units,), dtype=np.float32)

        if weapon_result.red_selected_target_idx.size > 0:
            for i, t in enumerate(weapon_result.red_selected_target_idx):
                if int(t) >= 0 and i < state.red.num_fighters:
                    if bool(weapon_result.red_selected_long[i]):
                        p = float(state.red.long_hit_prob[i])
                    elif bool(weapon_result.red_selected_short[i]):
                        p = float(state.red.short_hit_prob[i])
                    else:
                        p = 0.0
                    red_delta[i] += self.coef * p

        if weapon_result.blue_selected_target_idx.size > 0:
            for i, t in enumerate(weapon_result.blue_selected_target_idx):
                if int(t) >= 0 and i < state.blue.num_fighters:
                    if bool(weapon_result.blue_selected_long[i]):
                        p = float(state.blue.long_hit_prob[i])
                    elif bool(weapon_result.blue_selected_short[i]):
                        p = float(state.blue.short_hit_prob[i])
                    else:
                        p = 0.0
                    blue_delta[i] += self.coef * p

        return red_delta, blue_delta, {"coef": self.coef}
