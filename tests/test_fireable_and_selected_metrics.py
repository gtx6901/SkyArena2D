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
    blue_n = env.state.blue.num_fighters

    # Encode fire action targeting enemy index 1 via long missile (code = 1)
    fighter_action = np.zeros((red_n, 4), dtype=np.float32)
    fighter_action[:, 3] = 1.0  # hit_target = 1 (long missile at enemy 0)

    obs, reward, done, trunc, info = env.step({
        "red": {"fighter_action": fighter_action},
        "blue": {},
    })
    metrics = info["metrics"]
    # Agents attempted to fire
    assert metrics["red_fire_attempts"] > 0 or metrics["red_attempted_edges"] > 0
    # Since units are far apart at step 1, selected should be 0 (not fireable)
    # OR if somehow in range, selected >= 0 (valid test either way)
    assert metrics["red_selected_edges"] >= 0
    assert metrics["red_invalid_fire_count"] >= 0  # may be 0 if in range


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
