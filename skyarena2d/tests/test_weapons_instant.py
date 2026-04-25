from __future__ import annotations

import numpy as np

from skyarena2d.core.config import EnvConfig, FighterProfile
from skyarena2d.core.engine import SkyArenaEngine


def _base_config() -> EnvConfig:
    cfg = EnvConfig()
    cfg.teams.red_fighters = 1
    cfg.teams.blue_fighters = 1
    cfg.spawn.jitter = 0.0
    cfg.weapon.attack_effect_delay = 0
    cfg.weapon.hit_prob_enable = False
    cfg.dynamics.default_fighter_speed = 0.0
    cfg.radar.fighter_range = 500.0
    cfg.weapon.long_range = 500.0
    cfg.weapon.short_range = 500.0
    return cfg


def test_valid_fire_consumes_ammo() -> None:
    cfg = _base_config()
    env = SkyArenaEngine(cfg)
    env.reset(seed=0)
    state = env.get_state()
    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.blue.pos[0] = np.array([150.0, 100.0], dtype=np.float32)

    before = int(state.red.long_ammo[0])
    env.step(
        {
            "red": {"fighter_action": np.array([[0, 1, 0, 1]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
            "blue": {"fighter_action": np.array([[180, 1, 0, 0]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
        }
    )
    assert int(state.red.long_ammo[0]) == before - 1


def test_fire_out_of_range_invalid() -> None:
    cfg = _base_config()
    cfg.weapon.long_range = 50.0
    env = SkyArenaEngine(cfg)
    _, _ = env.reset(seed=0)
    state = env.get_state()
    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.blue.pos[0] = np.array([300.0, 100.0], dtype=np.float32)
    before = int(state.red.long_ammo[0])
    _, reward, _, _, _ = env.step(
        {
            "red": {"fighter_action": np.array([[0, 1, 0, 1]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
            "blue": {"fighter_action": np.array([[180, 1, 0, 0]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
        }
    )
    assert int(state.red.long_ammo[0]) == before
    assert reward["red"] < 0.0


def test_no_ammo_invalid() -> None:
    cfg = _base_config()
    env = SkyArenaEngine(cfg)
    env.reset(seed=0)
    state = env.get_state()
    state.red.long_ammo[0] = 0
    _, reward, _, _, _ = env.step(
        {
            "red": {"fighter_action": np.array([[0, 1, 0, 1]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
            "blue": {"fighter_action": np.array([[180, 1, 0, 0]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
        }
    )
    assert reward["red"] < 0.0


def test_hit_prob_disable_must_hit() -> None:
    cfg = _base_config()
    cfg.weapon.hit_prob_enable = False
    env = SkyArenaEngine(cfg)
    env.reset(seed=0)
    state = env.get_state()
    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.blue.pos[0] = np.array([120.0, 100.0], dtype=np.float32)
    env.step(
        {
            "red": {"fighter_action": np.array([[0, 1, 0, 1]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
            "blue": {"fighter_action": np.array([[180, 1, 0, 0]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
        }
    )
    assert not bool(state.blue.alive[0])


def test_simultaneous_multi_missile_resolution() -> None:
    cfg = EnvConfig()
    cfg.teams.red_fighters = 2
    cfg.teams.blue_fighters = 1
    cfg.spawn.jitter = 0.0
    cfg.weapon.attack_effect_delay = 0
    cfg.weapon.hit_prob_enable = True
    cfg.weapon.long_range = 500.0
    cfg.radar.fighter_range = 500.0
    cfg.dynamics.default_fighter_speed = 0.0
    cfg.red_fighter_profiles = [
        FighterProfile(long_hit_prob=0.0),
        FighterProfile(long_hit_prob=1.0),
    ]
    env = SkyArenaEngine(cfg)
    env.reset(seed=0)
    state = env.get_state()
    state.red.pos[:] = np.array([[100.0, 100.0], [100.0, 120.0]], dtype=np.float32)
    state.blue.pos[0] = np.array([130.0, 110.0], dtype=np.float32)

    env.step(
        {
            "red": {"fighter_action": np.array([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
            "blue": {"fighter_action": np.array([[180, 1, 0, 0]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
        }
    )
    assert not bool(state.blue.alive[0])
