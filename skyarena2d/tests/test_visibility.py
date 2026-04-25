from __future__ import annotations

import numpy as np

from skyarena2d.core.config import EnvConfig
from skyarena2d.core.engine import SkyArenaEngine
from skyarena2d.core.sensors import compute_visible_matrix


def _make_env_single(red_f: int = 1, red_d: int = 0, blue_f: int = 1, blue_d: int = 0) -> SkyArenaEngine:
    cfg = EnvConfig()
    cfg.teams.red_fighters = red_f
    cfg.teams.red_detectors = red_d
    cfg.teams.blue_fighters = blue_f
    cfg.teams.blue_detectors = blue_d
    cfg.spawn.mode = "fixed_scaled"
    cfg.spawn.jitter = 0.0
    cfg.max_steps = 20
    env = SkyArenaEngine(cfg)
    env.reset(seed=1)
    return env


def test_radar_off_not_visible() -> None:
    env = _make_env_single()
    state = env.get_state()
    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.blue.pos[0] = np.array([120.0, 100.0], dtype=np.float32)
    state.red.radar_on[0] = False
    state.red.radar_freq[0] = 0

    vis = compute_visible_matrix(state.red, state.blue).visible_matrix
    assert not bool(vis[0, 0])


def test_range_in_and_out_visibility() -> None:
    env = _make_env_single()
    state = env.get_state()
    state.red.radar_on[0] = True
    state.red.radar_freq[0] = 1
    state.red.radar_range[0] = 200.0
    state.red.radar_fov_deg[0] = 180.0
    state.red.heading[0] = 0.0

    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.blue.pos[0] = np.array([250.0, 100.0], dtype=np.float32)
    vis_in = compute_visible_matrix(state.red, state.blue).visible_matrix
    assert bool(vis_in[0, 0])

    state.blue.pos[0] = np.array([350.0, 100.0], dtype=np.float32)
    vis_out = compute_visible_matrix(state.red, state.blue).visible_matrix
    assert not bool(vis_out[0, 0])


def test_fov_outside_not_visible() -> None:
    env = _make_env_single()
    state = env.get_state()
    state.red.radar_on[0] = True
    state.red.radar_freq[0] = 1
    state.red.radar_range[0] = 300.0
    state.red.radar_fov_deg[0] = 60.0
    state.red.heading[0] = 0.0
    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.blue.pos[0] = np.array([100.0, 250.0], dtype=np.float32)

    vis = compute_visible_matrix(state.red, state.blue).visible_matrix
    assert not bool(vis[0, 0])


def test_detector_omni_visible() -> None:
    env = _make_env_single(red_f=0, red_d=1, blue_f=1, blue_d=0)
    state = env.get_state()
    state.red.radar_on[0] = True
    state.red.radar_freq[0] = 1
    state.red.radar_range[0] = 500.0
    state.red.radar_fov_deg[0] = 360.0
    state.red.heading[0] = 0.0
    state.red.pos[0] = np.array([300.0, 300.0], dtype=np.float32)
    state.blue.pos[0] = np.array([200.0, 200.0], dtype=np.float32)

    vis = compute_visible_matrix(state.red, state.blue).visible_matrix
    assert bool(vis[0, 0])
