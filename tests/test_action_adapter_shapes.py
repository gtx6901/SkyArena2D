"""Tests for SkyArenaActionAdapter shapes and behavior."""
from __future__ import annotations

import numpy as np
import pytest

from skyarena2d.core.state import TeamState
from skyarena2d.training.action_adapter import SkyArenaActionAdapter


def _make_team(n: int, side: str, pos_x_start: float = 0.0) -> TeamState:
    pos = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        pos[i, 0] = pos_x_start + i * 100.0
        pos[i, 1] = 500.0 + i * 50.0

    return TeamState(
        side=side,
        num_fighters=n,
        num_detectors=0,
        alive=np.ones(n, dtype=bool),
        pos=pos,
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


def _make_adapter(n_slots: int = 6) -> SkyArenaActionAdapter:
    return SkyArenaActionAdapter(
        candidate_slots=n_slots,
        course_bins=16,
        search_goal_grid_size=8,
        map_width=3000.0,
        map_height=4000.0,
        radar_freq=1,
        jammer_freq=0,
    )


def test_decode_output_shapes():
    """Verify SkyArenaSideAction has correct shapes for N=4."""
    N = 4
    own = _make_team(N, "red")
    adapter = _make_adapter()

    candidate_ids = np.full((N, 6), -1, dtype=np.int64)
    candidate_can_long = np.zeros((N, 6), dtype=bool)
    candidate_can_short = np.zeros((N, 6), dtype=bool)

    action = adapter.decode(
        course_action=np.zeros(N, dtype=np.int32),
        search_goal_action=np.zeros(N, dtype=np.int32),
        target_action=np.zeros(N, dtype=np.int32),
        fire_action=np.zeros(N, dtype=np.int32),
        own=own,
        candidate_ids=candidate_ids,
        candidate_can_long=candidate_can_long,
        candidate_can_short=candidate_can_short,
        has_active_contact=np.zeros(N, dtype=bool),
        current_heading=np.full(N, 90.0, dtype=np.float32),
    )

    assert action.course.shape == (N,), f"course shape: {action.course.shape}"
    assert action.radar_freq.shape == (N,), f"radar_freq shape: {action.radar_freq.shape}"
    assert action.jammer_freq.shape == (N,), f"jammer_freq shape: {action.jammer_freq.shape}"
    assert action.fire_type.shape == (N,), f"fire_type shape: {action.fire_type.shape}"
    assert action.target_idx.shape == (N,), f"target_idx shape: {action.target_idx.shape}"


def test_course_action_with_contact():
    """With has_active_contact=True, course should use course_action offset."""
    N = 1
    own = _make_team(N, "red")
    adapter = _make_adapter()

    candidate_ids = np.full((N, 6), -1, dtype=np.int64)
    candidate_can_long = np.zeros((N, 6), dtype=bool)
    candidate_can_short = np.zeros((N, 6), dtype=bool)

    # course_action=1 -> offset = +22.5 degrees (bin 1 of 16-bin LUT)
    action = adapter.decode(
        course_action=np.array([1], dtype=np.int32),
        search_goal_action=np.zeros(N, dtype=np.int32),
        target_action=np.zeros(N, dtype=np.int32),
        fire_action=np.zeros(N, dtype=np.int32),
        own=own,
        candidate_ids=candidate_ids,
        candidate_can_long=candidate_can_long,
        candidate_can_short=candidate_can_short,
        has_active_contact=np.array([True], dtype=bool),
        current_heading=np.array([90.0], dtype=np.float32),
    )

    # Expected: 90 + 22.5 = 112.5
    expected = (90.0 + 22.5) % 360.0
    assert abs(action.course[0] - expected) < 1e-3, (
        f"Expected course {expected}, got {action.course[0]}"
    )


def test_search_goal_action_without_contact():
    """Without contact, course should point toward search_goal region center."""
    N = 1
    own = _make_team(N, "red")
    adapter = _make_adapter()

    candidate_ids = np.full((N, 6), -1, dtype=np.int64)
    candidate_can_long = np.zeros((N, 6), dtype=bool)
    candidate_can_short = np.zeros((N, 6), dtype=bool)

    # Agent at (0, 500), search_goal_action=0 -> region 0 center
    # Region 0 center: col=0, row=0 -> cx = 0.5 * (3000/8) = 187.5, cy = 0.5 * (4000/8) = 250
    action = adapter.decode(
        course_action=np.zeros(N, dtype=np.int32),
        search_goal_action=np.array([0], dtype=np.int32),
        target_action=np.zeros(N, dtype=np.int32),
        fire_action=np.zeros(N, dtype=np.int32),
        own=own,
        candidate_ids=candidate_ids,
        candidate_can_long=candidate_can_long,
        candidate_can_short=candidate_can_short,
        has_active_contact=np.array([False], dtype=bool),
        current_heading=np.array([90.0], dtype=np.float32),
    )

    # Compute expected heading toward region 0 center
    cx, cy = adapter._region_centers[0]
    ox, oy = float(own.pos[0, 0]), float(own.pos[0, 1])
    expected = np.degrees(np.arctan2(cy - oy, cx - ox)) % 360.0

    assert abs(action.course[0] - expected) < 1e-3, (
        f"Expected course {expected}, got {action.course[0]}"
    )


def test_fire_blocked_by_mask():
    """Fire type 1 (long) should be blocked if candidate_can_long=False."""
    N = 1
    own = _make_team(N, "red")
    adapter = _make_adapter()

    # Slot 0 has enemy 0, but cannot fire long
    candidate_ids = np.array([[0, -1, -1, -1, -1, -1]], dtype=np.int64)
    candidate_can_long = np.array([[False, False, False, False, False, False]], dtype=bool)
    candidate_can_short = np.array([[False, False, False, False, False, False]], dtype=bool)

    action = adapter.decode(
        course_action=np.zeros(N, dtype=np.int32),
        search_goal_action=np.zeros(N, dtype=np.int32),
        target_action=np.array([1], dtype=np.int32),  # target slot 0
        fire_action=np.array([1], dtype=np.int32),    # try long fire
        own=own,
        candidate_ids=candidate_ids,
        candidate_can_long=candidate_can_long,
        candidate_can_short=candidate_can_short,
        has_active_contact=np.array([True], dtype=bool),
        current_heading=np.array([90.0], dtype=np.float32),
    )

    # Fire should be blocked
    assert action.fire_type[0] == 0, (
        f"Expected fire_type=0 (blocked), got {action.fire_type[0]}"
    )
    # Target idx should still be set
    assert action.target_idx[0] == 0, (
        f"Expected target_idx=0, got {action.target_idx[0]}"
    )


def test_target_action_zero_means_no_fire():
    """target_action=0 should result in target_idx=-1 and fire_type=0."""
    N = 2
    own = _make_team(N, "red")
    adapter = _make_adapter()

    candidate_ids = np.array([[0, 1, -1, -1, -1, -1], [0, -1, -1, -1, -1, -1]], dtype=np.int64)
    candidate_can_long = np.ones((N, 6), dtype=bool)
    candidate_can_short = np.ones((N, 6), dtype=bool)

    action = adapter.decode(
        course_action=np.zeros(N, dtype=np.int32),
        search_goal_action=np.zeros(N, dtype=np.int32),
        target_action=np.zeros(N, dtype=np.int32),  # 0 = no target
        fire_action=np.ones(N, dtype=np.int32),      # would fire long if target set
        own=own,
        candidate_ids=candidate_ids,
        candidate_can_long=candidate_can_long,
        candidate_can_short=candidate_can_short,
        has_active_contact=np.ones(N, dtype=bool),
        current_heading=np.full(N, 90.0, dtype=np.float32),
    )

    for i in range(N):
        assert action.target_idx[i] == -1, (
            f"Agent {i}: expected target_idx=-1, got {action.target_idx[i]}"
        )
        assert action.fire_type[i] == 0, (
            f"Agent {i}: expected fire_type=0, got {action.fire_type[i]}"
        )


def test_fire_allowed_when_can_long():
    """Fire type 1 (long) should be allowed when candidate_can_long=True."""
    N = 1
    own = _make_team(N, "red")
    adapter = _make_adapter()

    candidate_ids = np.array([[3, -1, -1, -1, -1, -1]], dtype=np.int64)
    candidate_can_long = np.array([[True, False, False, False, False, False]], dtype=bool)
    candidate_can_short = np.array([[False, False, False, False, False, False]], dtype=bool)

    action = adapter.decode(
        course_action=np.zeros(N, dtype=np.int32),
        search_goal_action=np.zeros(N, dtype=np.int32),
        target_action=np.array([1], dtype=np.int32),  # target slot 0
        fire_action=np.array([1], dtype=np.int32),    # long fire
        own=own,
        candidate_ids=candidate_ids,
        candidate_can_long=candidate_can_long,
        candidate_can_short=candidate_can_short,
        has_active_contact=np.array([True], dtype=bool),
        current_heading=np.array([90.0], dtype=np.float32),
    )

    assert action.fire_type[0] == 1, f"Expected fire_type=1, got {action.fire_type[0]}"
    assert action.target_idx[0] == 3, f"Expected target_idx=3, got {action.target_idx[0]}"


def test_build_masks_shapes():
    """Verify build_masks returns correct shapes."""
    N = 3
    own = _make_team(N, "red")
    adapter = _make_adapter()

    candidate_ids = np.full((N, 6), -1, dtype=np.int64)
    candidate_ids[0, 0] = 2  # agent 0 has one valid candidate

    fireable_long = np.zeros((N, 5), dtype=bool)
    fireable_short = np.zeros((N, 5), dtype=bool)

    masks = adapter.build_masks(own, fireable_long, fireable_short, candidate_ids)

    assert masks["course_mask"].shape == (N, 16)
    assert masks["search_goal_mask"].shape == (N, 64)
    assert masks["target_mask"].shape == (N, 7)

    # All alive agents should have all course bins enabled
    assert masks["course_mask"][0].all()

    # target_mask[0, 0] always True (no target option)
    assert masks["target_mask"][0, 0]
    # target_mask[0, 1] True because candidate_ids[0, 0] = 2 (valid)
    assert masks["target_mask"][0, 1]
    # target_mask[0, 2..] False because no more valid candidates
    assert not masks["target_mask"][0, 2]


def test_build_search_goal_lut():
    """Verify search goal LUT shape and values."""
    adapter = _make_adapter()
    lut = adapter.build_search_goal_lut()

    assert lut.shape == (64, 2), f"Expected (64, 2), got {lut.shape}"

    # First region center: col=0, row=0
    cell_w = 3000.0 / 8
    cell_h = 4000.0 / 8
    expected_cx = 0.5 * cell_w
    expected_cy = 0.5 * cell_h
    assert abs(lut[0, 0] - expected_cx) < 1e-3
    assert abs(lut[0, 1] - expected_cy) < 1e-3


def test_to_maca_fighter_action():
    """Verify SkyArenaSideAction.to_maca_fighter_action produces correct encoding."""
    N = 2
    own = _make_team(N, "red")
    adapter = _make_adapter()

    candidate_ids = np.array([[5, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]], dtype=np.int64)
    candidate_can_long = np.array([[True, False, False, False, False, False], [False] * 6], dtype=bool)
    candidate_can_short = np.array([[False] * 6, [False] * 6], dtype=bool)

    action = adapter.decode(
        course_action=np.zeros(N, dtype=np.int32),
        search_goal_action=np.zeros(N, dtype=np.int32),
        target_action=np.array([1, 0], dtype=np.int32),
        fire_action=np.array([1, 0], dtype=np.int32),
        own=own,
        candidate_ids=candidate_ids,
        candidate_can_long=candidate_can_long,
        candidate_can_short=candidate_can_short,
        has_active_contact=np.array([True, False], dtype=bool),
        current_heading=np.array([90.0, 45.0], dtype=np.float32),
    )

    maca = action.to_maca_fighter_action(max_enemy=10)
    assert maca.shape == (N, 4)
    # Agent 0 fires long at enemy 5: hit_target = 5 + 1 = 6
    assert maca[0, 3] == 6.0, f"Expected 6.0, got {maca[0, 3]}"
    # Agent 1 no fire
    assert maca[1, 3] == 0.0, f"Expected 0.0, got {maca[1, 3]}"


def test_rule_based_jammer_opens_on_contact_without_actor_head():
    N = 2
    own = _make_team(N, "red")
    adapter = SkyArenaActionAdapter(
        candidate_slots=6,
        course_bins=16,
        search_goal_grid_size=8,
        map_width=3000.0,
        map_height=4000.0,
        radar_freq=1,
        jammer_freq=1,
        use_jammer_strategy=True,
        jammer_range=320.0,
        max_jammers_per_side=1,
    )

    candidate_ids = np.array([[0, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1]], dtype=np.int64)
    candidate_can_long = np.zeros((N, 6), dtype=bool)
    candidate_can_short = np.zeros((N, 6), dtype=bool)
    entity_features = np.zeros((N, 6, 10), dtype=np.float32)
    diag = np.hypot(3000.0, 4000.0)
    entity_features[0, 0, 2] = 300.0 / diag
    entity_features[1, 0, 2] = 100.0 / diag

    action = adapter.decode(
        course_action=np.zeros(N, dtype=np.int32),
        search_goal_action=np.zeros(N, dtype=np.int32),
        target_action=np.zeros(N, dtype=np.int32),
        fire_action=np.zeros(N, dtype=np.int32),
        own=own,
        candidate_ids=candidate_ids,
        candidate_can_long=candidate_can_long,
        candidate_can_short=candidate_can_short,
        has_active_contact=np.ones(N, dtype=bool),
        current_heading=np.full(N, 90.0, dtype=np.float32),
        entity_features=entity_features,
        ew_state_key="test",
        step_count=5,
    )

    assert action.jammer_freq[0] == 0
    assert action.jammer_freq[1] == 1


def test_rule_based_jammer_memory_holds_then_expires():
    N = 1
    own = _make_team(N, "red")
    adapter = SkyArenaActionAdapter(
        candidate_slots=6,
        course_bins=16,
        search_goal_grid_size=8,
        map_width=3000.0,
        map_height=4000.0,
        radar_freq=1,
        jammer_freq=2,
        use_jammer_strategy=True,
        jammer_range=320.0,
        jammer_memory_steps=2,
    )

    candidate_ids = np.array([[0, -1, -1, -1, -1, -1]], dtype=np.int64)
    candidate_can_long = np.zeros((N, 6), dtype=bool)
    candidate_can_short = np.zeros((N, 6), dtype=bool)
    entity_features = np.zeros((N, 6, 10), dtype=np.float32)
    entity_features[0, 0, 2] = 100.0 / np.hypot(3000.0, 4000.0)

    adapter.decode(
        course_action=np.zeros(N, dtype=np.int32),
        search_goal_action=np.zeros(N, dtype=np.int32),
        target_action=np.zeros(N, dtype=np.int32),
        fire_action=np.zeros(N, dtype=np.int32),
        own=own,
        candidate_ids=candidate_ids,
        candidate_can_long=candidate_can_long,
        candidate_can_short=candidate_can_short,
        has_active_contact=np.ones(N, dtype=bool),
        current_heading=np.full(N, 90.0, dtype=np.float32),
        entity_features=entity_features,
        ew_state_key="test-memory",
        step_count=10,
    )

    no_candidates = np.full((N, 6), -1, dtype=np.int64)
    held = adapter.decode(
        course_action=np.zeros(N, dtype=np.int32),
        search_goal_action=np.zeros(N, dtype=np.int32),
        target_action=np.zeros(N, dtype=np.int32),
        fire_action=np.zeros(N, dtype=np.int32),
        own=own,
        candidate_ids=no_candidates,
        candidate_can_long=candidate_can_long,
        candidate_can_short=candidate_can_short,
        has_active_contact=np.zeros(N, dtype=bool),
        current_heading=np.full(N, 90.0, dtype=np.float32),
        entity_features=entity_features,
        ew_state_key="test-memory",
        step_count=12,
    )
    expired = adapter.decode(
        course_action=np.zeros(N, dtype=np.int32),
        search_goal_action=np.zeros(N, dtype=np.int32),
        target_action=np.zeros(N, dtype=np.int32),
        fire_action=np.zeros(N, dtype=np.int32),
        own=own,
        candidate_ids=no_candidates,
        candidate_can_long=candidate_can_long,
        candidate_can_short=candidate_can_short,
        has_active_contact=np.zeros(N, dtype=bool),
        current_heading=np.full(N, 90.0, dtype=np.float32),
        entity_features=entity_features,
        ew_state_key="test-memory",
        step_count=13,
    )

    assert held.jammer_freq[0] == 2
    assert expired.jammer_freq[0] == 0
