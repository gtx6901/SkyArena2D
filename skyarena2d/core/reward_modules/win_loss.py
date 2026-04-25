from __future__ import annotations

import numpy as np

from .base import RewardModule


class WinLossModule(RewardModule):
    """Terminal win/lose/draw reward. Only applied when termination_result.done == True."""

    def __init__(self, win: float = 5.0, lose: float = -5.0, draw: float = 0.0) -> None:
        self.win = win
        self.lose = lose
        self.draw = draw

    @property
    def name(self) -> str:
        return "win_loss"

    def compute(self, state, config, weapon_result, termination_result) -> tuple[np.ndarray, np.ndarray, dict]:
        red_delta = np.zeros((state.red.total_units,), dtype=np.float32)
        blue_delta = np.zeros((state.blue.total_units,), dtype=np.float32)

        if not termination_result.done:
            return red_delta, blue_delta, {}

        if termination_result.winner == "red":
            red_r, blue_r = self.win, self.lose
        elif termination_result.winner == "blue":
            red_r, blue_r = self.lose, self.win
        else:
            red_r, blue_r = self.draw, self.draw

        red_delta += red_r
        blue_delta += blue_r
        return red_delta, blue_delta, {"red": red_r, "blue": blue_r}
