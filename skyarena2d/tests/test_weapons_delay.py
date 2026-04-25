from __future__ import annotations

import numpy as np

from skyarena2d.core.config import EnvConfig
from skyarena2d.core.engine import SkyArenaEngine


def test_attack_effect_delay() -> None:
    cfg = EnvConfig()
    cfg.teams.red_fighters = 1
    cfg.teams.blue_fighters = 1
    cfg.spawn.jitter = 0.0
    cfg.weapon.attack_effect_delay = 2
    cfg.weapon.hit_prob_enable = False
    cfg.weapon.long_range = 500.0
    cfg.radar.fighter_range = 500.0
    cfg.dynamics.default_fighter_speed = 0.0

    env = SkyArenaEngine(cfg)
    env.reset(seed=1)
    state = env.get_state()
    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.blue.pos[0] = np.array([130.0, 100.0], dtype=np.float32)

    fire = {
        "red": {"fighter_action": np.array([[0, 1, 0, 1]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
        "blue": {"fighter_action": np.array([[180, 1, 0, 0]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
    }
    hold = {
        "red": {"fighter_action": np.array([[0, 1, 0, 0]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
        "blue": {"fighter_action": np.array([[180, 1, 0, 0]], dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
    }

    env.step(fire)
    assert bool(state.blue.alive[0])
    assert len(state.missile_queue) == 1

    env.step(hold)
    assert bool(state.blue.alive[0])

    env.step(hold)
    assert not bool(state.blue.alive[0])
