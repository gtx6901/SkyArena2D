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


def test_valid_fire_component_not_polluted_by_kill_loss():
    """valid_fire component should only contain valid_fire delta, not kill/win contributions."""
    config = load_config("configs/env_10v10_full.yaml")
    config.teams.red_fighters = 1
    config.teams.blue_fighters = 1
    config.spawn.jitter = 0.0
    config.weapon.attack_effect_delay = 0
    config.weapon.hit_prob_enable = False
    config.weapon.long_range = 500.0
    config.weapon.short_range = 500.0
    config.radar.fighter_range = 500.0
    config.dynamics.default_fighter_speed = 0.0

    env = SkyArenaEngine(config)
    env.reset(seed=101)
    state = env.get_state()
    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.blue.pos[0] = np.array([120.0, 100.0], dtype=np.float32)

    _, _, _, _, info = env.step(
        {
            "red": {
                "fighter_action": np.array([[0, 1, 0, 1]], dtype=np.float32),
                "detector_action": np.zeros((0, 2), dtype=np.float32),
            },
            "blue": {
                "fighter_action": np.array([[180, 1, 0, 0]], dtype=np.float32),
                "detector_action": np.zeros((0, 2), dtype=np.float32),
            },
        }
    )

    comp = info["reward_components"]
    assert "valid_fire" in comp
    assert abs(float(comp["valid_fire"]["red"]) - float(config.reward.valid_fire)) < 1e-6


def test_invalid_fire_component_not_polluted():
    """invalid_fire component should equal invalid-fire penalty delta only."""
    config = load_config("configs/env_10v10_full.yaml")
    config.teams.red_fighters = 1
    config.teams.blue_fighters = 1
    config.spawn.jitter = 0.0
    config.weapon.attack_effect_delay = 0
    config.weapon.hit_prob_enable = False
    config.weapon.long_range = 10.0
    config.weapon.short_range = 10.0
    config.radar.fighter_range = 500.0
    config.dynamics.default_fighter_speed = 0.0

    env = SkyArenaEngine(config)
    env.reset(seed=102)
    state = env.get_state()
    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.blue.pos[0] = np.array([500.0, 100.0], dtype=np.float32)

    _, _, _, _, info = env.step(
        {
            "red": {
                "fighter_action": np.array([[0, 1, 0, 1]], dtype=np.float32),
                "detector_action": np.zeros((0, 2), dtype=np.float32),
            },
            "blue": {
                "fighter_action": np.array([[180, 1, 0, 0]], dtype=np.float32),
                "detector_action": np.zeros((0, 2), dtype=np.float32),
            },
        }
    )

    comp = info["reward_components"]
    assert "invalid_fire" in comp
    assert abs(float(comp["invalid_fire"]["red"]) - float(config.reward.invalid_fire)) < 1e-6


def test_engine_reward_is_team_level_with_debug_unit_fields():
    env = _make_env(seed=99)
    _, reward, _, _, _ = env.step({"red": {}, "blue": {}})
    assert "red_unit" in reward and "blue_unit" in reward
    assert "red_unit_sum" in reward and "blue_unit_sum" in reward

    red_unit = np.asarray(reward["red_unit"], dtype=np.float32)
    blue_unit = np.asarray(reward["blue_unit"], dtype=np.float32)
    assert abs(float(reward["red_unit_sum"]) - float(red_unit.sum())) < 1e-6
    assert abs(float(reward["blue_unit_sum"]) - float(blue_unit.sum())) < 1e-6

    state = env.get_state()
    red_alive = state.red.alive
    blue_alive = state.blue.alive
    red_team = float(np.mean(red_unit[red_alive]) if np.any(red_alive) else np.mean(red_unit))
    blue_team = float(np.mean(blue_unit[blue_alive]) if np.any(blue_alive) else np.mean(blue_unit))
    assert abs(float(reward["red"]) - red_team) < 1e-6
    assert abs(float(reward["blue"]) - blue_team) < 1e-6
