from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RewardModule(ABC):
    @abstractmethod
    def compute(self, state, config, weapon_result, termination_result) -> tuple[np.ndarray, np.ndarray, dict]:
        """Returns (red_delta, blue_delta, component_info)"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
