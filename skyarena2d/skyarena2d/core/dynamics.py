from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DynamicsConfig
from .state import TeamState


@dataclass(slots=True)
class DynamicsResult:
    red_boundary_violations: int = 0
    blue_boundary_violations: int = 0


def heading_to_velocity(heading_deg: np.ndarray, speed: np.ndarray, dt: float) -> np.ndarray:
    theta = np.deg2rad(heading_deg)
    vel_x = np.cos(theta) * speed * dt
    vel_y = np.sin(theta) * speed * dt
    return np.stack([vel_x, vel_y], axis=1)


def apply_motion(
    team: TeamState,
    dynamics_cfg: DynamicsConfig,
    map_width: float,
    map_height: float,
    dt: float,
) -> int:
    if team.total_units == 0:
        return 0

    alive_mask = team.alive
    if not np.any(alive_mask):
        return 0

    deltas = heading_to_velocity(team.heading, team.speed, dt)
    team.pos[alive_mask] = team.pos[alive_mask] + deltas[alive_mask]

    out_left = team.pos[:, 0] < 0.0
    out_right = team.pos[:, 0] > map_width
    out_top = team.pos[:, 1] < 0.0
    out_bottom = team.pos[:, 1] > map_height
    out_any = (out_left | out_right | out_top | out_bottom) & alive_mask

    if not np.any(out_any):
        return 0

    mode = dynamics_cfg.boundary_mode
    if mode == "clamp":
        team.pos[:, 0] = np.clip(team.pos[:, 0], 0.0, map_width)
        team.pos[:, 1] = np.clip(team.pos[:, 1], 0.0, map_height)
    elif mode == "bounce":
        hit_vertical = (out_left | out_right) & alive_mask
        hit_horizontal = (out_top | out_bottom) & alive_mask
        team.heading[hit_vertical] = (180.0 - team.heading[hit_vertical]) % 360.0
        team.heading[hit_horizontal] = (-team.heading[hit_horizontal]) % 360.0
        team.pos[:, 0] = np.clip(team.pos[:, 0], 0.0, map_width)
        team.pos[:, 1] = np.clip(team.pos[:, 1], 0.0, map_height)
    elif mode == "kill":
        team.alive[out_any] = False
        team.pos[:, 0] = np.clip(team.pos[:, 0], 0.0, map_width)
        team.pos[:, 1] = np.clip(team.pos[:, 1], 0.0, map_height)
    elif mode == "penalty":
        team.pos[:, 0] = np.clip(team.pos[:, 0], 0.0, map_width)
        team.pos[:, 1] = np.clip(team.pos[:, 1], 0.0, map_height)

    return int(np.count_nonzero(out_any))
