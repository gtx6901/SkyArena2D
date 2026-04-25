from __future__ import annotations

import numpy as np

from skyarena2d.core.config import EnvConfig
from skyarena2d.core.engine import SkyArenaEngine
from skyarena2d.core.jamming import compute_jammed_matrix


def _env_for_jam() -> SkyArenaEngine:
    cfg = EnvConfig()
    cfg.teams.red_fighters = 1
    cfg.teams.blue_fighters = 1
    cfg.spawn.jitter = 0.0
    cfg.jamming.mode = "deterministic"
    env = SkyArenaEngine(cfg)
    env.reset(seed=2)
    state = env.get_state()
    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.blue.pos[0] = np.array([120.0, 100.0], dtype=np.float32)
    state.red.radar_on[0] = True
    state.red.radar_freq[0] = 3
    state.blue.jammer_on[0] = True
    return env


def test_spot_same_freq_blocks() -> None:
    env = _env_for_jam()
    state = env.get_state()
    state.blue.jammer_freq[0] = 3
    jam = compute_jammed_matrix(state.red, state.blue, env.config.radar, env.config.jamming, state.rng)
    assert bool(jam.jammed_matrix[0, 0])


def test_spot_diff_freq_not_block() -> None:
    env = _env_for_jam()
    state = env.get_state()
    state.blue.jammer_freq[0] = 2
    jam = compute_jammed_matrix(state.red, state.blue, env.config.radar, env.config.jamming, state.rng)
    assert not bool(jam.jammed_matrix[0, 0])


def test_barrage_blocks_in_deterministic_mode() -> None:
    env = _env_for_jam()
    state = env.get_state()
    state.blue.jammer_freq[0] = env.config.radar.freq_count + 1
    jam = compute_jammed_matrix(state.red, state.blue, env.config.radar, env.config.jamming, state.rng)
    assert bool(jam.jammed_matrix[0, 0])


def test_jam_out_of_range_no_effect() -> None:
    env = _env_for_jam()
    state = env.get_state()
    state.blue.jammer_freq[0] = 3
    state.blue.pos[0] = np.array([900.0, 900.0], dtype=np.float32)
    jam = compute_jammed_matrix(state.red, state.blue, env.config.radar, env.config.jamming, state.rng)
    assert not bool(jam.jammed_matrix[0, 0])
