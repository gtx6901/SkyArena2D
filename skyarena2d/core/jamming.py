from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import JammingConfig, RadarConfig
from .state import TeamState


@dataclass(slots=True)
class JammingResult:
    jammed_matrix: np.ndarray
    jammed_detection_count: int
    debug: dict[str, int]


def compute_jammed_matrix(
    own: TeamState,
    enemy: TeamState,
    radar_cfg: RadarConfig,
    jam_cfg: JammingConfig,
    rng: np.random.Generator,
) -> JammingResult:
    if own.total_units == 0 or enemy.total_units == 0 or not jam_cfg.enabled:
        shape = (own.total_units, enemy.total_units)
        return JammingResult(np.zeros(shape, dtype=bool), 0, {"spot": 0, "barrage": 0})

    jammer_mask = enemy.alive & enemy.jammer_on & (enemy.jammer_freq > 0)
    if not np.any(jammer_mask):
        shape = (own.total_units, enemy.total_units)
        return JammingResult(np.zeros(shape, dtype=bool), 0, {"spot": 0, "barrage": 0})

    jammer_idx = np.where(jammer_mask)[0]
    jammer_pos = enemy.pos[jammer_idx]
    jammer_freq = enemy.jammer_freq[jammer_idx]
    jammer_range = enemy.jammer_range[jammer_idx]

    dx = jammer_pos[None, :, 0] - own.pos[:, None, 0]
    dy = jammer_pos[None, :, 1] - own.pos[:, None, 1]
    d = np.sqrt(dx * dx + dy * dy)

    in_range = d <= jammer_range[None, :]
    own_radar_work = own.alive & own.radar_on & (own.radar_freq > 0)
    own_radar_work = own_radar_work[:, None]

    barrage_freq = radar_cfg.freq_count + 1
    spot_mask = in_range & own_radar_work & (own.radar_freq[:, None] == jammer_freq[None, :])
    barrage_mask = in_range & own_radar_work & (jammer_freq[None, :] == barrage_freq)

    if jam_cfg.mode == "deterministic":
        spot_effect = np.any(spot_mask, axis=1)
        barrage_effect = np.any(barrage_mask, axis=1) & (jam_cfg.barrage_block_prob > 0.0)
    else:
        spot_trials = rng.random(size=spot_mask.shape)
        barrage_trials = rng.random(size=barrage_mask.shape)
        spot_effect = np.any(spot_mask & (spot_trials < jam_cfg.spot_block_prob), axis=1)
        barrage_effect = np.any(barrage_mask & (barrage_trials < jam_cfg.barrage_block_prob), axis=1)

    jam_on_own = own.alive & (spot_effect | barrage_effect)
    jammed_matrix = jam_on_own[:, None] & enemy.alive[None, :]

    return JammingResult(
        jammed_matrix=jammed_matrix,
        jammed_detection_count=int(np.count_nonzero(jammed_matrix)),
        debug={
            "spot": int(np.count_nonzero(spot_mask)),
            "barrage": int(np.count_nonzero(barrage_mask)),
        },
    )
