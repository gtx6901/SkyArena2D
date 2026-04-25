from __future__ import annotations

import numpy as np

from skyarena2d.core.config import EnvConfig
from skyarena2d.core.engine import SkyArenaEngine
from skyarena2d.core.passive_detection import compute_passive_detection


def _setup() -> SkyArenaEngine:
    cfg = EnvConfig()
    cfg.teams.red_fighters = 1
    cfg.teams.blue_fighters = 1
    cfg.spawn.jitter = 0.0
    env = SkyArenaEngine(cfg)
    env.reset(seed=3)
    state = env.get_state()
    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.blue.pos[0] = np.array([150.0, 100.0], dtype=np.float32)
    return env


def test_enemy_radar_on_produces_j_recv() -> None:
    env = _setup()
    state = env.get_state()
    state.blue.radar_on[0] = True
    state.blue.radar_freq[0] = 2
    state.blue.jammer_on[0] = False
    recv, count = compute_passive_detection(state.red, state.blue, env.config.passive_detection)
    assert count >= 1
    assert recv[0][0]["id"] == 1


def test_enemy_jammer_on_produces_j_recv() -> None:
    env = _setup()
    state = env.get_state()
    state.blue.radar_on[0] = False
    state.blue.radar_freq[0] = 0
    state.blue.jammer_on[0] = True
    state.blue.jammer_freq[0] = 4
    recv, count = compute_passive_detection(state.red, state.blue, env.config.passive_detection)
    assert count >= 1
    assert recv[0][0]["r_fp"] == 4


def test_enemy_off_no_j_recv() -> None:
    env = _setup()
    state = env.get_state()
    state.blue.radar_on[0] = False
    state.blue.radar_freq[0] = 0
    state.blue.jammer_on[0] = False
    state.blue.jammer_freq[0] = 0
    recv, count = compute_passive_detection(state.red, state.blue, env.config.passive_detection)
    assert count == 0
    assert recv[0] == []
