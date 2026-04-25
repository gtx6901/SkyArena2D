from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import TeamState


@dataclass(slots=True)
class SensorResult:
    visible_matrix: np.ndarray
    distance_matrix: np.ndarray
    relative_bearing_matrix: np.ndarray


def compute_geometry(own: TeamState, enemy: TeamState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if own.total_units == 0 or enemy.total_units == 0:
        shape = (own.total_units, enemy.total_units)
        return (
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
        )

    dx = enemy.pos[None, :, 0] - own.pos[:, None, 0]
    dy = enemy.pos[None, :, 1] - own.pos[:, None, 1]
    distance = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
    absolute_bearing = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    relative = ((absolute_bearing - own.heading[:, None] + 540.0) % 360.0) - 180.0
    return distance, absolute_bearing.astype(np.float32), relative.astype(np.float32)


def compute_visible_matrix(
    own: TeamState,
    enemy: TeamState,
    jammed_matrix: np.ndarray | None = None,
) -> SensorResult:
    distance, _, relative = compute_geometry(own, enemy)

    if own.total_units == 0 or enemy.total_units == 0:
        return SensorResult(
            visible_matrix=np.zeros((own.total_units, enemy.total_units), dtype=bool),
            distance_matrix=distance,
            relative_bearing_matrix=relative,
        )

    alive_mask = own.alive[:, None] & enemy.alive[None, :]
    radar_work_mask = own.radar_on[:, None] & (own.radar_freq[:, None] > 0)
    range_mask = distance <= own.radar_range[:, None]
    fov = own.radar_fov_deg[:, None]
    omni_mask = fov >= 359.9
    fov_mask = omni_mask | (np.abs(relative) <= (fov * 0.5))

    visible = alive_mask & radar_work_mask & range_mask & fov_mask
    if jammed_matrix is not None:
        visible = visible & (~jammed_matrix)

    return SensorResult(
        visible_matrix=visible,
        distance_matrix=distance,
        relative_bearing_matrix=relative,
    )


def build_visible_list(
    enemy: TeamState,
    visible_row: np.ndarray,
    distance_row: np.ndarray,
    relative_row: np.ndarray,
) -> list[dict[str, float | int | str]]:
    visible_ids = np.where(visible_row)[0]
    out: list[dict[str, float | int | str]] = []
    for idx in visible_ids.tolist():
        out.append(
            {
                "id": idx + 1,
                "type": "fighter" if enemy.unit_type[idx] == 0 else "detector",
                "distance": float(distance_row[idx]),
                "direction": float(relative_row[idx]),
            }
        )
    return out
