from __future__ import annotations

import numpy as np

from .config import PassiveDetectionConfig
from .state import TeamState


def compute_passive_detection(
    own: TeamState,
    enemy: TeamState,
    passive_cfg: PassiveDetectionConfig,
) -> tuple[list[list[dict[str, float | int | str]]], int]:
    if not passive_cfg.enabled or own.total_units == 0 or enemy.total_units == 0:
        return ([[] for _ in range(own.total_units)], 0)

    dx = enemy.pos[None, :, 0] - own.pos[:, None, 0]
    dy = enemy.pos[None, :, 1] - own.pos[:, None, 1]
    dist = np.sqrt(dx * dx + dy * dy)

    absolute_bearing = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    relative_bearing = ((absolute_bearing - own.heading[:, None] + 540.0) % 360.0) - 180.0

    own_alive = own.alive[:, None]
    enemy_alive = enemy.alive[None, :]
    in_range = dist <= passive_cfg.range

    radar_emit = enemy.radar_on & (enemy.radar_freq > 0)
    jammer_emit = enemy.jammer_on & (enemy.jammer_freq > 0)
    emitter = np.zeros((enemy.total_units,), dtype=bool)
    if passive_cfg.detect_radar_on:
        emitter = emitter | radar_emit
    if passive_cfg.detect_jammer_on:
        emitter = emitter | jammer_emit

    matrix = own_alive & enemy_alive & in_range & emitter[None, :]

    recv: list[list[dict[str, float | int | str]]] = []
    total = 0
    for i in range(own.total_units):
        row_idx = np.where(matrix[i])[0]
        row: list[dict[str, float | int | str]] = []
        for j in row_idx.tolist():
            fp = int(enemy.radar_freq[j]) if radar_emit[j] else int(enemy.jammer_freq[j])
            row.append(
                {
                    "id": j + 1,
                    "type": "fighter" if enemy.unit_type[j] == 0 else "detector",
                    "direction": float(relative_bearing[i, j]),
                    "r_fp": fp,
                }
            )
        total += len(row)
        recv.append(row)

    return recv, total
