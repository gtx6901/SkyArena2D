from __future__ import annotations

import numpy as np

from skyarena2d.core.state import TeamState
from skyarena2d.training.action_adapter import TURN_DELTAS_DEG, SkyArenaActionAdapter


def _team(n: int = 2) -> TeamState:
    return TeamState(
        side="red",
        num_fighters=n,
        num_detectors=0,
        alive=np.ones(n, dtype=bool),
        pos=np.zeros((n, 2), dtype=np.float32),
        heading=np.full(n, 90.0, dtype=np.float32),
        speed=np.full(n, 2.0, dtype=np.float32),
        unit_type=np.zeros(n, dtype=np.int32),
        radar_on=np.ones(n, dtype=bool),
        radar_freq=np.ones(n, dtype=np.int32),
        jammer_on=np.zeros(n, dtype=bool),
        jammer_freq=np.zeros(n, dtype=np.int32),
        long_ammo=np.full(n, 2, dtype=np.int32),
        short_ammo=np.full(n, 4, dtype=np.int32),
        radar_range=np.full(n, 260.0, dtype=np.float32),
        radar_fov_deg=np.full(n, 120.0, dtype=np.float32),
        jammer_range=np.full(n, 300.0, dtype=np.float32),
        long_range=np.full(n, 220.0, dtype=np.float32),
        short_range=np.full(n, 120.0, dtype=np.float32),
        long_hit_prob=np.full(n, 0.35, dtype=np.float32),
        short_hit_prob=np.full(n, 0.60, dtype=np.float32),
        last_action=np.zeros((n, 4), dtype=np.float32),
        last_reward=np.zeros(n, dtype=np.float32),
    )


def _decode(adapter: SkyArenaActionAdapter, own: TeamState, **overrides):
    n, slots = own.num_fighters, adapter.candidate_slots
    args = {
        "course_action": np.full(n, 4, dtype=np.int64),
        "target_action": np.zeros(n, dtype=np.int64),
        "fire_action": np.zeros(n, dtype=np.int64),
        "own": own,
        "candidate_ids": np.full((n, slots), -1, dtype=np.int64),
        "candidate_can_long": np.zeros((n, slots), dtype=bool),
        "candidate_can_short": np.zeros((n, slots), dtype=bool),
        "has_active_contact": np.zeros(n, dtype=bool),
        "current_heading": own.heading[:n].copy(),
    }
    args.update(overrides)
    return adapter.decode(**args)


def test_atomic_turns_are_relative_to_current_heading() -> None:
    own = _team()
    adapter = SkyArenaActionAdapter(candidate_slots=3, use_jammer_strategy=False, jammer_freq=0)
    action = _decode(adapter, own, course_action=np.array([0, 8]))
    np.testing.assert_allclose(action.course, [0.0, 180.0])
    assert TURN_DELTAS_DEG.shape == (9,)


def test_pointer_can_select_only_enemy_tokens() -> None:
    own = _team(1)
    adapter = SkyArenaActionAdapter(candidate_slots=3, use_jammer_strategy=False, jammer_freq=0)
    ids = np.array([[-1, 4, -1]], dtype=np.int64)
    long = np.array([[False, True, False]])

    ally = _decode(
        adapter,
        own,
        target_action=np.array([1]),
        fire_action=np.array([1]),
        candidate_ids=ids,
        candidate_can_long=long,
    )
    enemy = _decode(
        adapter,
        own,
        target_action=np.array([2]),
        fire_action=np.array([1]),
        candidate_ids=ids,
        candidate_can_long=long,
    )
    assert ally.target_idx[0] == -1 and ally.fire_type[0] == 0
    assert enemy.target_idx[0] == 4 and enemy.fire_type[0] == 1


def test_weapon_choice_is_conditioned_on_selected_target() -> None:
    own = _team(1)
    adapter = SkyArenaActionAdapter(candidate_slots=2, use_jammer_strategy=False, jammer_freq=0)
    ids = np.array([[2, 3]])
    long = np.array([[True, False]])
    short = np.array([[False, True]])
    action = _decode(
        adapter,
        own,
        target_action=np.array([2]),
        fire_action=np.array([1]),
        candidate_ids=ids,
        candidate_can_long=long,
        candidate_can_short=short,
    )
    assert action.target_idx[0] == 3
    assert action.fire_type[0] == 0


def test_native_action_bridge_still_encodes_valid_launch() -> None:
    own = _team(1)
    adapter = SkyArenaActionAdapter(candidate_slots=1, use_jammer_strategy=False, jammer_freq=0)
    action = _decode(
        adapter,
        own,
        target_action=np.array([1]),
        fire_action=np.array([2]),
        candidate_ids=np.array([[5]]),
        candidate_can_short=np.array([[True]]),
    )
    encoded = action.to_maca_fighter_action(max_enemy=10)
    assert encoded.shape == (1, 4)
    assert encoded[0, 3] == 16.0


def test_dead_agent_keeps_heading_and_cannot_target() -> None:
    own = _team(1)
    own.alive[0] = False
    adapter = SkyArenaActionAdapter(candidate_slots=1, use_jammer_strategy=False, jammer_freq=0)
    action = _decode(
        adapter,
        own,
        course_action=np.array([8]),
        target_action=np.array([1]),
        fire_action=np.array([1]),
        candidate_ids=np.array([[0]]),
        candidate_can_long=np.array([[True]]),
    )
    assert action.course[0] == 90.0
    assert action.target_idx[0] == -1
    assert action.fire_type[0] == 0
