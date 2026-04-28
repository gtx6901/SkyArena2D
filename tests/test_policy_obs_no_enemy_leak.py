"""Tests for policy obs enemy leak prevention."""
from __future__ import annotations

import numpy as np
import pytest

from skyarena2d.core.state import TeamState
from skyarena2d.training.obs_builder import SkyArenaTrainingObsBuilder


def _make_team(n: int, side: str, pos_x_start: float = 0.0) -> TeamState:
    """Create a minimal TeamState with n fighters."""
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


def _make_builder(n: int = 2) -> SkyArenaTrainingObsBuilder:
    return SkyArenaTrainingObsBuilder(
        num_fighters=n,
        candidate_slots=6,
        search_goal_grid_size=8,
        map_width=3000.0,
        map_height=4000.0,
        semantic_map_size=32,
        track_memory_steps=10,
    )


def test_invisible_enemy_not_in_candidates():
    """Invisible enemy (visible_matrix=False, no track) must not appear in candidates."""
    red = _make_team(2, "red", pos_x_start=0.0)
    blue = _make_team(2, "blue", pos_x_start=1500.0)

    builder = _make_builder(2)

    # red[0] cannot see blue[0], red[0] can see blue[1]
    visible_matrix = np.array([[False, True], [True, True]], dtype=bool)
    fireable_long = np.zeros((2, 2), dtype=bool)
    fireable_short = np.zeros((2, 2), dtype=bool)

    obs = builder.build_policy_obs(
        own=red,
        enemy=blue,
        visible_matrix=visible_matrix,
        fireable_long=fireable_long,
        fireable_short=fireable_short,
        step_count=0,
        max_steps=1000,
        current_search_goal_id=np.zeros(2, dtype=np.int64),
    )

    candidate_ids = obs["candidate_ids"]  # (2, 6)
    entity_mask = obs["entity_mask"]      # (2, 6)

    # red[0] should NOT have blue[0] (index 0) in its candidates
    assert 0 not in candidate_ids[0], (
        f"Enemy 0 leaked into red[0] candidates: {candidate_ids[0]}"
    )

    # red[0] should have blue[1] (index 1) in its candidates
    assert 1 in candidate_ids[0], (
        f"Enemy 1 missing from red[0] candidates: {candidate_ids[0]}"
    )

    # entity_mask[0] should have exactly 1 True (for blue[1])
    assert entity_mask[0].sum() == 1, (
        f"Expected 1 valid candidate for red[0], got {entity_mask[0].sum()}"
    )


def test_visible_enemy_in_candidates():
    """Visible enemy must appear in candidates with entity_mask=True."""
    red = _make_team(2, "red", pos_x_start=0.0)
    blue = _make_team(2, "blue", pos_x_start=1500.0)

    builder = _make_builder(2)

    # red[0] can see blue[0]
    visible_matrix = np.array([[True, False], [False, False]], dtype=bool)
    fireable_long = np.zeros((2, 2), dtype=bool)
    fireable_short = np.zeros((2, 2), dtype=bool)

    obs = builder.build_policy_obs(
        own=red,
        enemy=blue,
        visible_matrix=visible_matrix,
        fireable_long=fireable_long,
        fireable_short=fireable_short,
        step_count=0,
        max_steps=1000,
        current_search_goal_id=np.zeros(2, dtype=np.int64),
    )

    candidate_ids = obs["candidate_ids"]
    entity_mask = obs["entity_mask"]

    # red[0] should have blue[0] (index 0) in its candidates
    assert 0 in candidate_ids[0], (
        f"Enemy 0 missing from red[0] candidates: {candidate_ids[0]}"
    )

    # Find the slot for blue[0]
    slot = np.where(candidate_ids[0] == 0)[0]
    assert len(slot) == 1
    assert entity_mask[0, slot[0]], "entity_mask should be True for visible enemy"


def test_global_state_contains_all_enemies():
    """Global state must contain all enemy positions regardless of visibility."""
    red = _make_team(2, "red", pos_x_start=0.0)
    blue = _make_team(2, "blue", pos_x_start=1500.0)

    builder = _make_builder(2)

    global_state = builder.build_global_state(
        red=red,
        blue=blue,
        step_count=5,
        max_steps=1000,
    )

    assert global_state.shape == (181,), f"Expected (181,), got {global_state.shape}"
    assert global_state.dtype == np.float32

    # step_progress at index 0
    assert abs(global_state[0] - 5 / 1000) < 1e-5

    # Blue positions should be encoded (non-zero) regardless of visibility
    # Blue starts at index 1 + 10*9 = 91
    blue_section = global_state[91:91 + 2 * 9]
    # blue[0] pos_x = 1500 / 3000 = 0.5
    assert abs(blue_section[0] - 0.5) < 1e-4, (
        f"Blue[0] pos_x not encoded correctly: {blue_section[0]}"
    )


def test_semantic_map_no_invisible_enemy():
    """enemy_visible channel (channel 3) must be 0 for invisible enemies."""
    red = _make_team(2, "red", pos_x_start=0.0)
    blue = _make_team(2, "blue", pos_x_start=1500.0)

    builder = _make_builder(2)

    # red[0] cannot see blue[0]
    visible_matrix = np.array([[False, False], [False, False]], dtype=bool)
    fireable_long = np.zeros((2, 2), dtype=bool)
    fireable_short = np.zeros((2, 2), dtype=bool)

    obs = builder.build_policy_obs(
        own=red,
        enemy=blue,
        visible_matrix=visible_matrix,
        fireable_long=fireable_long,
        fireable_short=fireable_short,
        step_count=0,
        max_steps=1000,
        current_search_goal_id=np.zeros(2, dtype=np.int64),
    )

    semantic_map = obs["semantic_map"]  # (N, 9, M, M)
    # Channel 3 = enemy_visible
    enemy_visible_channel = semantic_map[:, 3, :, :]

    # Should be all zeros since no enemies are visible
    assert enemy_visible_channel.sum() == 0.0, (
        f"enemy_visible channel has non-zero values for invisible enemies: "
        f"{enemy_visible_channel.sum()}"
    )


def test_obs_shapes_10v10():
    """Verify all observation shapes for a 10v10 scenario."""
    N = 10
    red = _make_team(N, "red", pos_x_start=0.0)
    blue = _make_team(N, "blue", pos_x_start=1500.0)

    builder = SkyArenaTrainingObsBuilder(
        num_fighters=N,
        candidate_slots=6,
        search_goal_grid_size=8,
        map_width=3000.0,
        map_height=4000.0,
        semantic_map_size=100,
        track_memory_steps=10,
    )

    visible_matrix = np.eye(N, dtype=bool)  # each agent sees one enemy
    fireable_long = np.zeros((N, N), dtype=bool)
    fireable_short = np.zeros((N, N), dtype=bool)

    obs = builder.build_policy_obs(
        own=red,
        enemy=blue,
        visible_matrix=visible_matrix,
        fireable_long=fireable_long,
        fireable_short=fireable_short,
        step_count=100,
        max_steps=1500,
        current_search_goal_id=np.zeros(N, dtype=np.int64),
    )

    assert obs["self_features"].shape == (N, 20)
    assert obs["entity_features"].shape == (N, 6, 10)
    assert obs["entity_mask"].shape == (N, 6)
    assert obs["candidate_ids"].shape == (N, 6)
    assert obs["candidate_can_long"].shape == (N, 6)
    assert obs["candidate_can_short"].shape == (N, 6)
    assert obs["semantic_map"].shape == (N, 9, 100, 100)
    assert obs["current_search_goal_id"].shape == (N,)
    assert obs["agent_id"].shape == (N,)
    assert obs["region_features"].shape == (N, 64, 10)
    assert obs["alive_mask"].shape == (N,)
    assert obs["has_active_contact"].shape == (N,)
    assert obs["course_mask"].shape == (N, 32)
    assert obs["search_goal_mask"].shape == (N, 64)
    assert obs["target_mask"].shape == (N, 7)


def test_track_memory_includes_recently_seen():
    """Enemy seen in previous step should still appear in candidates via track memory."""
    red = _make_team(1, "red", pos_x_start=0.0)
    blue = _make_team(1, "blue", pos_x_start=500.0)

    builder = _make_builder(1)

    # Step 0: enemy is visible
    visible_step0 = np.array([[True]], dtype=bool)
    fireable_long = np.zeros((1, 1), dtype=bool)
    fireable_short = np.zeros((1, 1), dtype=bool)

    builder.build_policy_obs(
        own=red,
        enemy=blue,
        visible_matrix=visible_step0,
        fireable_long=fireable_long,
        fireable_short=fireable_short,
        step_count=0,
        max_steps=1000,
        current_search_goal_id=np.zeros(1, dtype=np.int64),
    )

    # Step 5: enemy is no longer visible, but within track_memory_steps=10
    visible_step5 = np.array([[False]], dtype=bool)
    obs = builder.build_policy_obs(
        own=red,
        enemy=blue,
        visible_matrix=visible_step5,
        fireable_long=fireable_long,
        fireable_short=fireable_short,
        step_count=5,
        max_steps=1000,
        current_search_goal_id=np.zeros(1, dtype=np.int64),
    )

    # Enemy should still appear via track memory
    candidate_ids = obs["candidate_ids"]
    assert 0 in candidate_ids[0], (
        f"Tracked enemy missing from candidates at step 5: {candidate_ids[0]}"
    )


def test_track_memory_expires():
    """Enemy not seen for > track_memory_steps should be excluded."""
    red = _make_team(1, "red", pos_x_start=0.0)
    blue = _make_team(1, "blue", pos_x_start=500.0)

    builder = _make_builder(1)

    # Step 0: enemy is visible
    visible_step0 = np.array([[True]], dtype=bool)
    fireable_long = np.zeros((1, 1), dtype=bool)
    fireable_short = np.zeros((1, 1), dtype=bool)

    builder.build_policy_obs(
        own=red,
        enemy=blue,
        visible_matrix=visible_step0,
        fireable_long=fireable_long,
        fireable_short=fireable_short,
        step_count=0,
        max_steps=1000,
        current_search_goal_id=np.zeros(1, dtype=np.int64),
    )

    # Step 15: enemy not visible, beyond track_memory_steps=10
    visible_step15 = np.array([[False]], dtype=bool)
    obs = builder.build_policy_obs(
        own=red,
        enemy=blue,
        visible_matrix=visible_step15,
        fireable_long=fireable_long,
        fireable_short=fireable_short,
        step_count=15,
        max_steps=1000,
        current_search_goal_id=np.zeros(1, dtype=np.int64),
    )

    candidate_ids = obs["candidate_ids"]
    assert 0 not in candidate_ids[0], (
        f"Expired track memory leaked enemy into candidates: {candidate_ids[0]}"
    )
