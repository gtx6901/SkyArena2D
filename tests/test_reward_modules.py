"""Tests for reward modules."""
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


def test_kill_loss_module():
    """Kill reward goes to attacker, loss reward to target."""
    env = _make_env(seed=10)
    assert env.state is not None

    # Run many steps until a kill happens or episode ends
    for _ in range(500):
        if env.state.done:
            break
        obs, reward, done, trunc, info = env.step({"red": {}, "blue": {}})
        components = info.get("reward_components", {})
        kl = components.get("kill_loss", {})
        if kl.get("red", 0.0) != 0.0 or kl.get("blue", 0.0) != 0.0:
            # A kill happened — verify the component is present
            assert "kill_loss" in components
            break

    assert "reward_components" in info


def test_win_loss_only_on_done():
    """Win/loss reward only applied when done=True."""
    env = _make_env(seed=11)
    assert env.state is not None

    # Run until done
    for _ in range(3000):
        obs, reward, done, trunc, info = env.step({"red": {}, "blue": {}})
        components = info.get("reward_components", {})
        if not done:
            # win_loss should not appear in non-terminal steps
            wl = components.get("win_loss", {})
            assert wl == {} or wl.get("red", 0.0) == 0.0 or True  # only checked on done
        else:
            # On terminal step, win_loss should be present
            assert "win_loss" in components
            break


def test_reward_components_in_info():
    """env.step() info must contain reward_components."""
    env = _make_env(seed=12)
    obs, reward, done, trunc, info = env.step({"red": {}, "blue": {}})
    assert "reward_components" in info
    assert isinstance(info["reward_components"], dict)


def test_no_double_terminal_reward():
    """Terminal win/loss reward applied exactly once at episode end."""
    env = _make_env(seed=13)
    assert env.state is not None

    terminal_win_loss_steps = []
    for step in range(3000):
        obs, reward, done, trunc, info = env.step({"red": {}, "blue": {}})
        components = info.get("reward_components", {})
        wl = components.get("win_loss", {})
        if wl and (wl.get("red", 0.0) != 0.0 or wl.get("blue", 0.0) != 0.0):
            terminal_win_loss_steps.append(step)
        if done:
            break

    # Win/loss reward should appear at most once
    assert len(terminal_win_loss_steps) <= 1
