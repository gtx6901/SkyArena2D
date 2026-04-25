from __future__ import annotations

from typing import Any

import numpy as np

from ..core.sensors import build_visible_list
from ..core.state import TeamState


def _build_strike_views(
    *,
    side: str,
    unit_idx: int,
    strike_list: list[dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]]]:
    outgoing: list[dict[str, Any]] = []
    incoming: list[dict[str, Any]] = []
    for item in strike_list:
        if item.get("attacker_side") == side and int(item.get("attacker_idx", -1)) == unit_idx:
            outgoing.append(item)
        if item.get("target_side") == side and int(item.get("target_idx", -1)) == unit_idx:
            incoming.append(item)

    striking_list = [int(x.get("target_idx", -1)) + 1 for x in outgoing if "target_idx" in x]
    striking_dict_list = outgoing + incoming
    return striking_list, striking_dict_list


def build_maca_raw_obs(
    *,
    own: TeamState,
    enemy: TeamState,
    visible_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    bearing_matrix: np.ndarray,
    passive_lists: list[list[dict[str, Any]]],
    strike_list: list[dict[str, Any]],
) -> dict[str, Any]:
    fighter_obs_list: list[dict[str, Any]] = []
    detector_obs_list: list[dict[str, Any]] = []

    passive_enemy_set: set[int] = set()
    for row in passive_lists:
        for item in row:
            passive_enemy_set.add(int(item["id"]))

    for idx in range(own.num_fighters):
        striking_list, striking_dict_list = _build_strike_views(
            side=own.side,
            unit_idx=idx,
            strike_list=strike_list,
        )
        fighter_obs_list.append(
            {
                "id": idx + 1,
                "alive": bool(own.alive[idx]),
                "pos_x": float(own.pos[idx, 0]),
                "pos_y": float(own.pos[idx, 1]),
                "course": float(own.heading[idx]),
                "r_iswork": bool(own.radar_on[idx]),
                "r_fre_point": int(own.radar_freq[idx]),
                "r_visible_list": build_visible_list(
                    enemy,
                    visible_matrix[idx],
                    distance_matrix[idx],
                    bearing_matrix[idx],
                ),
                "j_iswork": bool(own.jammer_on[idx]),
                "j_fre_point": int(own.jammer_freq[idx]),
                "j_recv_list": passive_lists[idx],
                "l_missile_left": int(own.long_ammo[idx]),
                "s_missile_left": int(own.short_ammo[idx]),
                "striking_list": striking_list,
                "striking_dict_list": striking_dict_list,
                "last_reward": float(own.last_reward[idx]),
                "last_action": own.last_action[idx].tolist(),
            }
        )

    for local_det_idx, idx in enumerate(range(own.num_fighters, own.total_units)):
        detector_obs_list.append(
            {
                "id": idx + 1,
                "alive": bool(own.alive[idx]),
                "pos_x": float(own.pos[idx, 0]),
                "pos_y": float(own.pos[idx, 1]),
                "course": float(own.heading[idx]),
                "r_iswork": bool(own.radar_on[idx]),
                "r_fre_point": int(own.radar_freq[idx]),
                "r_visible_list": build_visible_list(
                    enemy,
                    visible_matrix[idx],
                    distance_matrix[idx],
                    bearing_matrix[idx],
                ),
                "last_reward": float(own.last_reward[idx]),
                "last_action": own.last_action[idx][:2].tolist(),
            }
        )
        _ = local_det_idx

    return {
        "detector_obs_list": detector_obs_list,
        "fighter_obs_list": fighter_obs_list,
        "joint_obs_dict": {
            "strike_list": strike_list,
            "passive_detection_enemy_list": sorted(passive_enemy_set),
        },
    }
