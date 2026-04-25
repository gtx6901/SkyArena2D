from __future__ import annotations

import numpy as np

from ..core.state import TeamState


def _normalize_angle_deg(angle: np.ndarray) -> np.ndarray:
    return ((angle + 180.0) % 360.0 - 180.0) / 180.0


def _safe_div(x: np.ndarray, denom: float) -> np.ndarray:
    if denom <= 0:
        return x.astype(np.float32)
    return (x / denom).astype(np.float32)


def build_modern_obs(
    own: TeamState,
    enemy: TeamState,
    visible_matrix: np.ndarray,
    fireable_long: np.ndarray,
    fireable_short: np.ndarray,
    map_width: float,
    map_height: float,
    freq_count: int,
) -> dict[str, object]:
    n = own.total_units
    m = enemy.total_units
    max_allies = max(0, n - 1)

    if n == 0:
        return {
            "self": np.zeros((0, 14), dtype=np.float32),
            "allies": np.zeros((0, 0, 10), dtype=np.float32),
            "enemies": np.zeros((0, m, 13), dtype=np.float32),
            "masks": {
                "ally": np.zeros((0, 0), dtype=bool),
                "enemy": np.zeros((0, m), dtype=bool),
                "self_alive": np.zeros((0,), dtype=bool),
            },
            "global_state": np.zeros((12,), dtype=np.float32),
            "visible_matrix": visible_matrix.astype(np.float32),
            "fireable_long": fireable_long.astype(np.float32),
            "fireable_short": fireable_short.astype(np.float32),
        }

    speed_norm = max(1.0, float(max(np.max(own.speed), np.max(enemy.speed)) if m > 0 else np.max(own.speed)))
    long_ammo_norm = max(1.0, float(np.max(own.long_ammo) if n > 0 else 1.0))
    short_ammo_norm = max(1.0, float(np.max(own.short_ammo) if n > 0 else 1.0))

    self_feats = np.stack(
        [
            _safe_div(own.pos[:, 0], map_width),
            _safe_div(own.pos[:, 1], map_height),
            (own.heading % 360.0) / 360.0,
            _safe_div(own.speed, speed_norm),
            own.alive.astype(np.float32),
            (own.unit_type == 0).astype(np.float32),
            (own.unit_type == 1).astype(np.float32),
            _safe_div(own.long_ammo.astype(np.float32), long_ammo_norm),
            _safe_div(own.short_ammo.astype(np.float32), short_ammo_norm),
            own.radar_on.astype(np.float32),
            _safe_div(own.radar_freq.astype(np.float32), max(1, freq_count)),
            own.jammer_on.astype(np.float32),
            _safe_div(own.jammer_freq.astype(np.float32), max(1, freq_count + 1)),
            own.last_reward.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    ally_feats = np.zeros((n, max_allies, 10), dtype=np.float32)
    ally_mask = np.zeros((n, max_allies), dtype=bool)
    for i in range(n):
        allies = [j for j in range(n) if j != i]
        if not allies:
            continue
        ally_pos = own.pos[np.array(allies, dtype=np.int64)]
        dx = ally_pos[:, 0] - own.pos[i, 0]
        dy = ally_pos[:, 1] - own.pos[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        bearing = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        rel = ((bearing - own.heading[i] + 540.0) % 360.0) - 180.0
        rows = len(allies)
        ally_mask[i, :rows] = True
        ally_feats[i, :rows, :] = np.stack(
            [
                _safe_div(dx, map_width),
                _safe_div(dy, map_height),
                _safe_div(dist, np.hypot(map_width, map_height)),
                _normalize_angle_deg(rel),
                own.alive[np.array(allies, dtype=np.int64)].astype(np.float32),
                (own.unit_type[np.array(allies, dtype=np.int64)] == 0).astype(np.float32),
                _safe_div(own.speed[np.array(allies, dtype=np.int64)], speed_norm),
                own.radar_on[np.array(allies, dtype=np.int64)].astype(np.float32),
                own.jammer_on[np.array(allies, dtype=np.int64)].astype(np.float32),
                own.last_reward[np.array(allies, dtype=np.int64)].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)

    dx_e = enemy.pos[None, :, 0] - own.pos[:, None, 0] if m > 0 else np.zeros((n, 0), dtype=np.float32)
    dy_e = enemy.pos[None, :, 1] - own.pos[:, None, 1] if m > 0 else np.zeros((n, 0), dtype=np.float32)
    dist_e = np.sqrt(dx_e * dx_e + dy_e * dy_e) if m > 0 else np.zeros((n, 0), dtype=np.float32)
    bearing_e = (np.degrees(np.arctan2(dy_e, dx_e)) + 360.0) % 360.0 if m > 0 else np.zeros((n, 0), dtype=np.float32)
    rel_e = ((bearing_e - own.heading[:, None] + 540.0) % 360.0) - 180.0 if m > 0 else np.zeros((n, 0), dtype=np.float32)

    enemy_feats = np.zeros((n, m, 13), dtype=np.float32)
    if m > 0:
        enemy_feats[:, :, 0] = _safe_div(dx_e, map_width)
        enemy_feats[:, :, 1] = _safe_div(dy_e, map_height)
        enemy_feats[:, :, 2] = _safe_div(dist_e, np.hypot(map_width, map_height))
        enemy_feats[:, :, 3] = _normalize_angle_deg(rel_e)
        enemy_feats[:, :, 4] = enemy.alive[None, :].astype(np.float32)
        enemy_feats[:, :, 5] = (enemy.unit_type[None, :] == 0).astype(np.float32)
        enemy_feats[:, :, 6] = _safe_div(enemy.speed[None, :], speed_norm)
        enemy_feats[:, :, 7] = enemy.radar_on[None, :].astype(np.float32)
        enemy_feats[:, :, 8] = enemy.jammer_on[None, :].astype(np.float32)
        enemy_feats[:, :, 9] = visible_matrix.astype(np.float32)
        enemy_feats[:, :, 10] = fireable_long.astype(np.float32)
        enemy_feats[:, :, 11] = fireable_short.astype(np.float32)
        enemy_feats[:, :, 12] = _safe_div(enemy.radar_freq[None, :].astype(np.float32), max(1, freq_count))

    enemy_mask = np.broadcast_to(enemy.alive[None, :], (n, m)).copy() if m > 0 else np.zeros((n, 0), dtype=bool)

    global_state = np.array(
        [
            own.alive_count,
            enemy.alive_count,
            own.fighter_alive_count,
            enemy.fighter_alive_count,
            own.detector_alive_count,
            enemy.detector_alive_count,
            float(np.mean(own.long_ammo) if n > 0 else 0.0),
            float(np.mean(enemy.long_ammo) if m > 0 else 0.0),
            float(np.mean(own.short_ammo) if n > 0 else 0.0),
            float(np.mean(enemy.short_ammo) if m > 0 else 0.0),
            float(np.count_nonzero(visible_matrix)),
            float(np.count_nonzero(fireable_long | fireable_short)),
        ],
        dtype=np.float32,
    )

    return {
        "self": self_feats,
        "allies": ally_feats,
        "enemies": enemy_feats,
        "masks": {
            "ally": ally_mask,
            "enemy": enemy_mask,
            "self_alive": own.alive.copy(),
        },
        "global_state": global_state,
        "visible_matrix": visible_matrix.astype(np.float32),
        "fireable_long": fireable_long.astype(np.float32),
        "fireable_short": fireable_short.astype(np.float32),
    }
