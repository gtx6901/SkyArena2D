"""Regression tests for fixed-scale team credit."""

from __future__ import annotations

import numpy as np

from skyarena2d.core.config import EnvConfig
from skyarena2d.core.engine import SkyArenaEngine
from skyarena2d.core.reward import compute_rewards
from skyarena2d.core.termination import TerminationResult
from skyarena2d.core.weapons import WeaponStepResult


def _state_2v2():
    config = EnvConfig()
    config.teams.red_fighters = 2
    config.teams.blue_fighters = 2
    engine = SkyArenaEngine(config)
    engine.reset(seed=7)
    return config, engine.get_state()


def _weapon_result(
    *,
    red_invalid: np.ndarray | None = None,
    blue_invalid: np.ndarray | None = None,
    resolved_records: list[dict[str, object]] | None = None,
) -> WeaponStepResult:
    red_invalid = np.zeros(2, dtype=bool) if red_invalid is None else red_invalid
    blue_invalid = np.zeros(2, dtype=bool) if blue_invalid is None else blue_invalid
    return WeaponStepResult(
        red_fireable_long=np.zeros((2, 2), dtype=bool),
        red_fireable_short=np.zeros((2, 2), dtype=bool),
        blue_fireable_long=np.zeros((2, 2), dtype=bool),
        blue_fireable_short=np.zeros((2, 2), dtype=bool),
        launch_records=[],
        resolved_records=[] if resolved_records is None else resolved_records,
        red_valid_fire=np.zeros(2, dtype=bool),
        red_invalid_fire=red_invalid,
        blue_valid_fire=np.zeros(2, dtype=bool),
        blue_invalid_fire=blue_invalid,
        killed_red=[],
        killed_blue=[],
        missiles_launched_long=0,
        missiles_launched_short=0,
        missiles_hit=0,
        missiles_missed=0,
    )


def _not_done() -> TerminationResult:
    return TerminationResult(done=False, truncated=False, winner="ongoing", reason="")


def test_team_reward_uses_fixed_roster_denominator_and_keeps_destroyed_loss() -> None:
    config, state = _state_2v2()
    state.red.alive[1] = False
    weapon_result = _weapon_result(
        resolved_records=[
            {
                "target_destroyed": True,
                "target_side": "red",
                "target_idx": 1,
                "incoming": [{"attacker_idx": 0}],
            }
        ]
    )

    output = compute_rewards(
        state=state,
        config=config,
        weapon_result=weapon_result,
        termination_result=_not_done(),
    )

    expected_loss = float(config.reward.loss_fighter)
    assert np.isclose(output.red_unit_rewards[1], expected_loss)
    assert np.isclose(output.red_team_reward, expected_loss / state.red.total_units)


def test_invalid_fire_penalty_is_red_blue_symmetric_and_applied_once() -> None:
    config, state = _state_2v2()
    red_invalid = np.array([True, False])
    blue_invalid = np.array([True, False])

    output = compute_rewards(
        state=state,
        config=config,
        weapon_result=_weapon_result(red_invalid=red_invalid, blue_invalid=blue_invalid),
        termination_result=_not_done(),
    )

    penalty = float(config.reward.invalid_fire)
    np.testing.assert_allclose(output.red_unit_rewards, [penalty, 0.0])
    np.testing.assert_allclose(output.blue_unit_rewards, [penalty, 0.0])
    assert np.isclose(output.red_team_reward, penalty / state.red.total_units)
    assert np.isclose(output.blue_team_reward, penalty / state.blue.total_units)
    assert np.isclose(output.components["invalid_fire"]["red"], penalty)
    assert np.isclose(output.components["invalid_fire"]["blue"], penalty)
