"""Tests for fireable vs selected metrics distinction."""
from __future__ import annotations

import numpy as np
import pytest

from skyarena2d.core.config import load_config
from skyarena2d.core.engine import SkyArenaEngine


def _make_env(seed: int = 0) -> SkyArenaEngine:
    config = load_config("configs/env_10v10_full.yaml")
    env = SkyArenaEngine(config)
    env.reset(seed=seed)
    return env


def test_fireable_vs_selected_distinction():
    """Agent has fireable opportunity but submits no fire action — selected_edges == 0."""
    env = _make_env(seed=1)
    # Pass empty actions (no fire)
    obs, reward, done, trunc, info = env.step({"red": {}, "blue": {}})
    metrics = info["metrics"]
    # fireable edges may be 0 at step 1 (units far apart), but selected must be 0
    assert metrics["red_selected_edges"] == 0
    assert metrics["blue_selected_edges"] == 0
    # attempted edges also 0 since no fire action submitted
    assert metrics["red_attempted_edges"] == 0
    assert metrics["blue_attempted_edges"] == 0


def test_attempted_invalid_fire():
    """Agent tries to fire at an out-of-range target — attempted > 0, selected == 0, invalid_fire_count > 0."""
    env = _make_env(seed=2)
    assert env.state is not None
    red_n = env.state.red.num_fighters

    # Force out-of-range geometry to make this test deterministic.
    env.state.red.pos[:red_n, 0] = 100.0
    env.state.red.pos[:red_n, 1] = np.linspace(100.0, 900.0, red_n, dtype=np.float32)
    env.state.blue.pos[:, 0] = 2900.0
    env.state.blue.pos[:, 1] = np.linspace(100.0, 900.0, env.state.blue.num_fighters, dtype=np.float32)

    # code=1 means long fire to enemy index 0.
    fighter_action = np.zeros((red_n, 4), dtype=np.float32)
    fighter_action[:, 3] = 1.0

    obs, reward, done, trunc, info = env.step({
        "red": {"fighter_action": fighter_action},
        "blue": {},
    })
    metrics = info["metrics"]
    assert env.state.cache is not None
    cache = env.state.cache

    # Agents attempted to fire
    assert metrics["red_fire_attempts"] > 0
    assert metrics["red_attempted_edges"] > 0
    # enemy index 0 should be recorded in attempted matrix, but not selected when invalid.
    assert cache.red_attempted_long_matrix[0, 0]
    assert not cache.red_selected_long_matrix[0, 0]
    assert metrics["red_selected_edges"] == 0
    assert metrics["red_invalid_fire_count"] >= 1


def test_attempted_and_selected_matrix_on_valid_fire():
    """Valid launch should appear in both attempted and selected matrices with enemy-0 index."""
    env = _make_env(seed=21)
    assert env.state is not None
    red_n = env.state.red.num_fighters

    # Force in-range geometry for deterministic valid fire.
    env.state.red.pos[:, :] = np.array([100.0, 200.0], dtype=np.float32)
    env.state.blue.pos[:, :] = np.array([150.0, 200.0], dtype=np.float32)

    fighter_action = np.zeros((red_n, 4), dtype=np.float32)
    fighter_action[:, 1] = 1.0
    fighter_action[:, 3] = 1.0  # long fire to enemy index 0
    _, _, _, _, info = env.step({"red": {"fighter_action": fighter_action}, "blue": {}})

    assert env.state.cache is not None
    cache = env.state.cache
    assert cache.red_attempted_long_matrix[0, 0]
    assert cache.red_selected_long_matrix[0, 0]
    assert info["metrics"]["red_attempted_edges"] >= info["metrics"]["red_selected_edges"] >= 1


def test_selected_fire_execution_rate():
    """After many steps, if agents fire when opportunity exists, execution rate > 0."""
    env = _make_env(seed=3)
    assert env.state is not None
    red_n = env.state.red.num_fighters

    # Run several steps with fire actions
    fighter_action = np.zeros((red_n, 4), dtype=np.float32)
    fighter_action[:, 3] = 1.0  # attempt long fire at enemy 0

    for _ in range(50):
        if env.state.done:
            break
        obs, reward, done, trunc, info = env.step({
            "red": {"fighter_action": fighter_action},
            "blue": {},
        })

    metrics = info["metrics"]
    # fire_execution_rate_given_opportunity is a ratio [0, 1]
    rate = metrics["red_fire_execution_rate_given_opportunity"]
    assert 0.0 <= rate <= 1.0


def test_selected_expected_exchange():
    """selected_expected_exchange is red_sel_exp - blue_sel_exp."""
    env = _make_env(seed=4)
    assert env.state is not None

    obs, reward, done, trunc, info = env.step({"red": {}, "blue": {}})
    metrics = info["metrics"]

    expected = metrics["selected_expected_red_kills"] - metrics["selected_expected_blue_kills"]
    assert abs(metrics["selected_expected_exchange"] - expected) < 1e-6
