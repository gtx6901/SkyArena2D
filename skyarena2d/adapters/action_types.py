from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SkyArenaSideAction:
    """Native SkyArena action for one side.

    The training pipeline uses this type as its internal action contract.
    Conversion to MaCA-style fighter_action exists only as a legacy bridge for
    current engine compatibility.
    """
    course: np.ndarray        # (num_fighters,) float32, absolute heading degrees [0, 360)
    radar_freq: np.ndarray    # (num_fighters,) int32, 0=off, 1..freq_count=on
    jammer_freq: np.ndarray   # (num_fighters,) int32, 0=off, 1..freq_count+1=on
    fire_type: np.ndarray     # (num_fighters,) int32, 0=no fire, 1=long, 2=short
    target_idx: np.ndarray    # (num_fighters,) int32, enemy index (0-based), -1=none

    def to_maca_fighter_action(self, max_enemy: int) -> np.ndarray:
        """Convert to legacy MaCA fighter_action array (N, 4) for engine compatibility.

        This conversion is a compatibility bridge, not a design goal to mirror
        the complete MaCA API surface.
        """
        n = len(self.course)
        arr = np.zeros((n, 4), dtype=np.float32)
        arr[:, 0] = self.course
        arr[:, 1] = self.radar_freq
        arr[:, 2] = self.jammer_freq
        for i in range(n):
            t = int(self.target_idx[i])
            ft = int(self.fire_type[i])
            if t >= 0 and ft == 1:
                arr[i, 3] = t + 1              # long: 1-indexed
            elif t >= 0 and ft == 2:
                arr[i, 3] = t + 1 + max_enemy  # short: offset by max_enemy
        return arr
