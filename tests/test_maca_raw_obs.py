from __future__ import annotations

import numpy as np

from skyarena2d.core.config import EnvConfig
from skyarena2d.core.engine import SkyArenaEngine


def test_maca_raw_obs_fields_and_lists() -> None:
    cfg = EnvConfig()
    cfg.teams.red_fighters = 2
    cfg.teams.blue_fighters = 2
    cfg.spawn.jitter = 0.0
    cfg.dynamics.default_fighter_speed = 0.0
    cfg.weapon.attack_effect_delay = 0
    cfg.weapon.long_range = 500.0
    cfg.radar.fighter_range = 500.0

    env = SkyArenaEngine(cfg)
    obs, _ = env.reset(seed=0)
    state = env.get_state()
    state.red.pos[:] = np.array([[100.0, 100.0], [120.0, 120.0]], dtype=np.float32)
    state.blue.pos[:] = np.array([[140.0, 100.0], [160.0, 120.0]], dtype=np.float32)

    obs, _, _, _, _ = env.step(
        {
            "red": {"fighter_action": np.array([[0, 1, 0, 1], [0, 1, 0, 0]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
            "blue": {"fighter_action": np.array([[180, 1, 0, 0], [180, 1, 0, 0]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
        }
    )

    raw = obs["red"]["raw"]
    assert "fighter_obs_list" in raw
    assert "detector_obs_list" in raw
    assert "joint_obs_dict" in raw

    f0 = raw["fighter_obs_list"][0]
    required = {
        "id",
        "alive",
        "pos_x",
        "pos_y",
        "course",
        "r_iswork",
        "r_fre_point",
        "r_visible_list",
        "j_iswork",
        "j_fre_point",
        "j_recv_list",
        "l_missile_left",
        "s_missile_left",
        "striking_list",
        "striking_dict_list",
        "last_reward",
        "last_action",
    }
    assert required.issubset(set(f0.keys()))
    assert isinstance(f0["r_visible_list"], list)
    assert isinstance(f0["j_recv_list"], list)
    assert isinstance(f0["striking_list"], list)

    if f0["r_visible_list"]:
        vis_item = f0["r_visible_list"][0]
        assert "id" in vis_item and "type" in vis_item
    if f0["j_recv_list"]:
        recv_item = f0["j_recv_list"][0]
        assert "id" in recv_item and "direction" in recv_item and "r_fp" in recv_item
