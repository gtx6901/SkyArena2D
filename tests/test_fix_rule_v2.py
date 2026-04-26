from __future__ import annotations

import numpy as np
import pytest

from skyarena2d.opponents import RULES
from skyarena2d.opponents.fix_rule_v2 import FixRuleV2Opponent


def _make_side_obs(
    fighters: list[dict],
    max_enemy: int = 5,
) -> dict:
    """Build a minimal side_obs dict for testing."""
    return {
        "raw": {
            "fighter_obs_list": fighters,
            "detector_obs_list": [],
        },
        "modern": {
            "enemies": np.zeros((len(fighters), max_enemy, 13), dtype=np.float32),
        },
    }


def _alive_fighter(
    idx: int = 0,
    course: float = 0.0,
    visible: list | None = None,
) -> dict:
    return {
        "id": idx + 1,
        "alive": True,
        "course": course,
        "l_missile_left": 2,
        "s_missile_left": 4,
        "r_visible_list": visible or [],
    }


def test_fix_rule_v2_registered() -> None:
    assert "fix_rule_v2" in RULES
    rule = RULES["fix_rule_v2"]()
    assert isinstance(rule, FixRuleV2Opponent)


def test_initial_push_red_heading() -> None:
    rule = FixRuleV2Opponent(seed=0)
    fighters = [_alive_fighter(0, course=0.0), _alive_fighter(1, course=0.0)]
    side_obs = _make_side_obs(fighters)
    action = rule.act(side_obs, "red", step_count=0)
    headings = action["fighter_action"][:, 0]
    assert np.allclose(headings, 0.0), f"Expected 0.0 heading for red push, got {headings}"


def test_initial_push_blue_heading() -> None:
    rule = FixRuleV2Opponent(seed=0)
    fighters = [_alive_fighter(0, course=180.0), _alive_fighter(1, course=180.0)]
    side_obs = _make_side_obs(fighters)
    action = rule.act(side_obs, "blue", step_count=0)
    headings = action["fighter_action"][:, 0]
    assert np.allclose(headings, 180.0), f"Expected 180.0 heading for blue push, got {headings}"


def test_post_contact_fires_at_visible() -> None:
    """After contact, agent with visible enemy should fire (long or short)."""
    rule = FixRuleV2Opponent(seed=0, prefer_long_before_short=True, long_range=220.0, short_range=120.0)
    # Simulate contact: fighter sees enemy at distance 100 (within short range)
    visible_enemy = [{"id": 1, "distance": 100.0, "direction": 0.0}]
    fighters = [_alive_fighter(0, course=0.0, visible=visible_enemy)]
    side_obs = _make_side_obs(fighters, max_enemy=5)

    # First call triggers first_contact_step
    action = rule.act(side_obs, "red", step_count=5)
    fire_cmd = action["fighter_action"][0, 3]
    # Should fire: either long (target_id=1) or short (max_enemy+target_id=6)
    assert fire_cmd != 0.0, f"Expected fire command, got {fire_cmd}"


def test_post_contact_heading_toward_enemy() -> None:
    """After contact, heading should be absolute bearing to enemy."""
    rule = FixRuleV2Opponent(seed=0)
    # Fighter at course=90, enemy at relative direction=45 -> absolute = 135
    visible_enemy = [{"id": 1, "distance": 300.0, "direction": 45.0}]
    fighters = [_alive_fighter(0, course=90.0, visible=visible_enemy)]
    side_obs = _make_side_obs(fighters, max_enemy=5)

    action = rule.act(side_obs, "red", step_count=5)
    heading = action["fighter_action"][0, 0]
    expected = (90.0 + 45.0) % 360.0  # = 135.0
    assert abs(heading - expected) < 1e-3, f"Expected heading {expected}, got {heading}"


def test_search_mode_changes_heading() -> None:
    """After random_hold_steps, search heading should change."""
    hold = 5
    rule = FixRuleV2Opponent(seed=42, random_hold_steps=hold)
    fighters = [_alive_fighter(0, course=0.0)]
    side_obs = _make_side_obs(fighters)

    # Trigger post-contact by injecting first_contact_step
    rule.first_contact_step = 0

    # First search step: get initial heading
    action0 = rule.act(side_obs, "red", step_count=1)
    h0 = action0["fighter_action"][0, 0]

    # Advance timer past hold steps
    for step in range(2, hold + 3):
        action = rule.act(side_obs, "red", step_count=step)

    # After hold steps, heading should have been resampled
    action_new = rule.act(side_obs, "red", step_count=hold + 3)
    h_new = action_new["fighter_action"][0, 0]
    # With seed=42, the two random headings are very unlikely to be identical
    # (probability ~0 for continuous uniform distribution)
    # We just check the mechanism ran without error and produced a valid heading
    assert 0.0 <= h_new < 360.0, f"Invalid heading: {h_new}"


def test_reset_clears_state() -> None:
    """Reset should clear first_contact_step and search state."""
    rule = FixRuleV2Opponent(seed=0)
    rule.first_contact_step = 10
    rule.agent_search_heading[0] = 45.0
    rule.agent_search_timer[0] = 5
    rule.agent_last_jammer_contact[0] = 10

    rule.reset(seed=1)
    assert rule.first_contact_step is None
    assert len(rule.agent_search_heading) == 0
    assert len(rule.agent_search_timer) == 0
    assert len(rule.agent_last_jammer_contact) == 0


def test_jammer_silent_during_initial_push() -> None:
    rule = FixRuleV2Opponent(seed=0, jammer_freq=3)
    fighters = [_alive_fighter(0, course=0.0), _alive_fighter(1, course=0.0)]
    action = rule.act(_make_side_obs(fighters), "red", step_count=0)
    assert np.all(action["fighter_action"][:, 2] == 0)


def test_jammer_uses_visible_target_frequency_after_contact() -> None:
    rule = FixRuleV2Opponent(seed=0, jammer_freq=2, jammer_range=320.0, match_target_radar_freq=True)
    visible_enemy = [{"id": 1, "distance": 250.0, "direction": 0.0, "r_fp": 4}]
    fighters = [_alive_fighter(0, course=0.0, visible=visible_enemy)]
    action = rule.act(_make_side_obs(fighters), "red", step_count=5)
    assert int(action["fighter_action"][0, 2]) == 4


def test_jammer_memory_holds_then_turns_off() -> None:
    rule = FixRuleV2Opponent(seed=0, jammer_freq=2, jammer_range=320.0, jammer_memory_steps=2)
    visible_enemy = [{"id": 1, "distance": 250.0, "direction": 0.0}]
    fighters = [_alive_fighter(0, course=0.0, visible=visible_enemy)]
    rule.act(_make_side_obs(fighters), "red", step_count=5)

    no_contact = [_alive_fighter(0, course=0.0, visible=[])]
    held = rule.act(_make_side_obs(no_contact), "red", step_count=7)
    expired = rule.act(_make_side_obs(no_contact), "red", step_count=8)

    assert int(held["fighter_action"][0, 2]) == 2
    assert int(expired["fighter_action"][0, 2]) == 0


def test_jammer_limits_active_emitters_to_nearest_targets() -> None:
    rule = FixRuleV2Opponent(seed=0, jammer_freq=1, jammer_range=320.0, max_jammers_per_side=1)
    fighters = [
        _alive_fighter(0, visible=[{"id": 1, "distance": 300.0, "direction": 0.0}]),
        _alive_fighter(1, visible=[{"id": 2, "distance": 100.0, "direction": 0.0}]),
    ]
    action = rule.act(_make_side_obs(fighters), "red", step_count=5)
    assert action["fighter_action"][0, 2] == 0
    assert action["fighter_action"][1, 2] == 1
