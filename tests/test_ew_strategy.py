"""Tests for training-side EW heuristic frequency selection."""
from __future__ import annotations

import numpy as np

from skyarena2d.core.state import TeamState
from skyarena2d.training.action_adapter import SkyArenaActionAdapter
from skyarena2d.training.ew_strategy import EWHeuristicStrategy


def _make_team(n: int) -> TeamState:
    return TeamState(
        side="red",
        num_fighters=n,
        num_detectors=0,
        alive=np.ones(n, dtype=bool),
        pos=np.zeros((n, 2), dtype=np.float32),
        heading=np.zeros(n, dtype=np.float32),
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


def _empty_candidates(n: int, slots: int = 6) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((n, slots), -1, dtype=np.int64),
        np.zeros((n, slots, 10), dtype=np.float32),
    )


def test_radar_freq_hops_and_stays_in_range() -> None:
    n = 4
    strategy = EWHeuristicStrategy(
        radar_freq_count=10,
        radar_cycle_interval=8,
        radar_stride=3,
    )
    candidate_ids, entity_features = _empty_candidates(n)
    alive = np.ones(n, dtype=bool)
    contact = np.zeros(n, dtype=bool)

    radar_0, _ = strategy.compute(
        key="radar-hop",
        n_agents=n,
        alive=alive,
        candidate_ids=candidate_ids,
        has_active_contact=contact,
        entity_features=entity_features,
        step_count=0,
    )
    radar_8, _ = strategy.compute(
        key="radar-hop",
        n_agents=n,
        alive=alive,
        candidate_ids=candidate_ids,
        has_active_contact=contact,
        entity_features=entity_features,
        step_count=8,
    )

    assert not np.array_equal(radar_0, radar_8)
    assert np.all((radar_0 >= 1) & (radar_0 <= 10))
    assert np.all((radar_8 >= 1) & (radar_8 <= 10))


def test_dead_agent_radar_freq_is_zero() -> None:
    n = 3
    strategy = EWHeuristicStrategy(radar_freq_count=10)
    candidate_ids, entity_features = _empty_candidates(n)
    alive = np.array([True, False, True], dtype=bool)

    radar_freq, _ = strategy.compute(
        key="dead-agent",
        n_agents=n,
        alive=alive,
        candidate_ids=candidate_ids,
        has_active_contact=np.zeros(n, dtype=bool),
        entity_features=entity_features,
        step_count=3,
    )

    assert radar_freq[1] == 0
    assert radar_freq[0] > 0
    assert radar_freq[2] > 0


def test_no_contact_keeps_jammer_off() -> None:
    n = 4
    strategy = EWHeuristicStrategy(radar_freq_count=10)
    candidate_ids, entity_features = _empty_candidates(n)

    _, jammer_freq = strategy.compute(
        key="no-contact",
        n_agents=n,
        alive=np.ones(n, dtype=bool),
        candidate_ids=candidate_ids,
        has_active_contact=np.zeros(n, dtype=bool),
        entity_features=entity_features,
        step_count=4,
    )

    assert np.array_equal(jammer_freq, np.zeros(n, dtype=np.int32))


def test_contact_selects_at_most_top_k_jammers_in_range() -> None:
    n = 5
    strategy = EWHeuristicStrategy(
        radar_freq_count=10,
        jammer_range=320.0,
        max_jammers_per_side=2,
    )
    candidate_ids = np.full((n, 6), -1, dtype=np.int64)
    candidate_ids[:, 0] = 0
    entity_features = np.zeros((n, 6, 10), dtype=np.float32)
    diag = np.hypot(3000.0, 4000.0)
    distances = np.array([50.0, 120.0, 250.0, 500.0, 80.0], dtype=np.float32)
    entity_features[:, 0, 2] = distances / diag

    _, jammer_freq = strategy.compute(
        key="top-k",
        n_agents=n,
        alive=np.ones(n, dtype=bool),
        candidate_ids=candidate_ids,
        has_active_contact=np.ones(n, dtype=bool),
        entity_features=entity_features,
        step_count=12,
    )

    assert np.count_nonzero(jammer_freq) <= 2
    assert jammer_freq[0] > 0
    assert jammer_freq[4] > 0


def test_compute_jammer_freq_compatibility() -> None:
    n = 2
    strategy = EWHeuristicStrategy(jammer_range=320.0)
    candidate_ids = np.full((n, 6), -1, dtype=np.int64)
    candidate_ids[:, 0] = 0
    entity_features = np.zeros((n, 6, 10), dtype=np.float32)
    entity_features[:, 0, 2] = 100.0 / np.hypot(3000.0, 4000.0)

    jammer_freq = strategy.compute_jammer_freq(
        key="compat",
        n_agents=n,
        alive=np.ones(n, dtype=bool),
        candidate_ids=candidate_ids,
        has_active_contact=np.ones(n, dtype=bool),
        entity_features=entity_features,
        step_count=6,
    )

    assert jammer_freq.shape == (n,)
    assert np.count_nonzero(jammer_freq) > 0


def test_action_adapter_uses_ew_strategy_radar_freq() -> None:
    n = 3
    own = _make_team(n)
    adapter = SkyArenaActionAdapter(
        candidate_slots=6,
        course_bins=16,
        search_goal_grid_size=8,
        map_width=3000.0,
        map_height=4000.0,
        radar_freq=1,
        radar_freq_count=10,
        radar_cycle_interval=8,
        radar_stride=3,
        use_jammer_strategy=True,
    )
    candidate_ids, entity_features = _empty_candidates(n)

    action = adapter.decode(
        course_action=np.zeros(n, dtype=np.int32),
        search_goal_action=np.zeros(n, dtype=np.int32),
        target_action=np.zeros(n, dtype=np.int32),
        fire_action=np.zeros(n, dtype=np.int32),
        own=own,
        candidate_ids=candidate_ids,
        candidate_can_long=np.zeros((n, 6), dtype=bool),
        candidate_can_short=np.zeros((n, 6), dtype=bool),
        has_active_contact=np.zeros(n, dtype=bool),
        current_heading=np.zeros(n, dtype=np.float32),
        entity_features=entity_features,
        ew_state_key="adapter-radar",
        step_count=8,
    )

    assert action.radar_freq.tolist() == [2, 5, 8]
    assert not np.all(action.radar_freq == 1)
