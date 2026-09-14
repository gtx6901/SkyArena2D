from __future__ import annotations

import numpy as np
import torch

from skyarena2d.core.state import TeamState
from skyarena2d.rl.models.actor import SkyArenaActor
from skyarena2d.rl.models.encoders import EntityEncoder
from skyarena2d.training.obs_builder import (
    ENTITY_FEATURE_DIM,
    SELF_FEATURE_DIM,
    SkyArenaTrainingObsBuilder,
)


def _team(
    positions: list[tuple[float, float]],
    *,
    side: str,
    headings: list[float] | None = None,
) -> TeamState:
    n = len(positions)
    heading = headings if headings is not None else [0.0] * n
    return TeamState(
        side=side,
        num_fighters=n,
        num_detectors=0,
        alive=np.ones(n, dtype=bool),
        pos=np.asarray(positions, dtype=np.float32),
        heading=np.asarray(heading, dtype=np.float32),
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


def _policy_obs(
    builder: SkyArenaTrainingObsBuilder,
    own: TeamState,
    enemy: TeamState,
    visible: np.ndarray,
    *,
    step: int = 0,
    fire_long: np.ndarray | None = None,
    fire_short: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    shape = visible.shape
    return builder.build_policy_obs(
        own=own,
        enemy=enemy,
        visible_matrix=visible,
        fireable_long=np.zeros(shape, dtype=bool) if fire_long is None else fire_long,
        fireable_short=np.zeros(shape, dtype=bool) if fire_short is None else fire_short,
        step_count=step,
        max_steps=100,
    )


def test_compact_obs_contains_all_allies_and_enemy_slots_with_strict_target_mask() -> None:
    own = _team([(100, 100), (200, 100), (300, 100)], side="red")
    enemy = _team([(500, 100), (600, 100), (700, 100)], side="blue")
    builder = SkyArenaTrainingObsBuilder(num_fighters=3, candidate_slots=1)
    visible = np.asarray(
        [[True, False, True], [False, False, False], [False, True, False]],
        dtype=bool,
    )
    obs = _policy_obs(builder, own, enemy, visible)

    # Two ally slots plus max(candidate_slots, num_fighters)==three enemy slots.
    assert obs["self_features"].shape == (3, SELF_FEATURE_DIM)
    assert obs["entity_features"].shape == (3, 5, ENTITY_FEATURE_DIM)
    assert obs["entity_mask"][0].tolist() == [True, True, True, False, True]
    assert obs["candidate_ids"][0].tolist() == [-1, -1, 0, -1, 2]
    assert obs["target_mask"][0].tolist() == [True, False, False, True, False, True]
    assert obs["course_mask"].shape == (3, 9)
    assert "semantic_map" not in obs
    assert "region_features" not in obs
    assert "current_search_goal_id" not in obs
    assert "search_goal_mask" not in obs


def test_entity_coordinates_are_translation_invariant_and_heading_aligned() -> None:
    visible = np.ones((2, 2), dtype=bool)
    base_own = _team([(100, 200), (200, 200)], side="red", headings=[0, 0])
    base_enemy = _team([(300, 250), (400, 100)], side="blue", headings=[30, 45])
    base = _policy_obs(
        SkyArenaTrainingObsBuilder(num_fighters=2), base_own, base_enemy, visible
    )["entity_features"][0]

    translated_own = _team([(600, 900), (700, 900)], side="red", headings=[0, 0])
    translated_enemy = _team([(800, 950), (900, 800)], side="blue", headings=[30, 45])
    translated = _policy_obs(
        SkyArenaTrainingObsBuilder(num_fighters=2),
        translated_own,
        translated_enemy,
        visible,
    )["entity_features"][0]
    np.testing.assert_allclose(base, translated, atol=1e-7)

    # Rotate the whole scene +90 degrees about the observing fighter and rotate
    # every heading +90 degrees: egocentric tokens must remain unchanged.
    rotated_own = _team([(100, 200), (100, 300)], side="red", headings=[90, 90])
    rotated_enemy = _team([(50, 400), (200, 500)], side="blue", headings=[120, 135])
    rotated = _policy_obs(
        SkyArenaTrainingObsBuilder(num_fighters=2),
        rotated_own,
        rotated_enemy,
        visible,
    )["entity_features"][0]
    np.testing.assert_allclose(base, rotated, atol=1e-7)


def test_recent_track_is_selectable_but_not_fireable_and_invisible_never_leaks() -> None:
    own = _team([(100, 100)], side="red")
    enemy = _team([(300, 100)], side="blue")
    builder = SkyArenaTrainingObsBuilder(
        num_fighters=1, candidate_slots=1, track_memory_steps=10
    )
    fire_long = np.ones((1, 1), dtype=bool)
    visible = np.ones((1, 1), dtype=bool)
    first = _policy_obs(
        builder, own, enemy, visible, fire_long=fire_long
    )
    assert first["target_mask"][0].tolist() == [True, True]
    assert first["candidate_can_long"][0].tolist() == [True]

    tracked = _policy_obs(
        builder, own, enemy, np.zeros((1, 1), dtype=bool), step=5
    )
    assert tracked["target_mask"][0].tolist() == [True, True]
    assert tracked["candidate_can_long"][0].tolist() == [False]
    assert tracked["entity_features"][0, 0, 14] == 1.0
    assert tracked["entity_features"][0, 0, 15] == 0.5

    fresh = _policy_obs(
        SkyArenaTrainingObsBuilder(num_fighters=1, candidate_slots=1),
        own,
        enemy,
        np.zeros((1, 1), dtype=bool),
    )
    assert not fresh["entity_mask"].any()
    assert fresh["candidate_ids"][0].tolist() == [-1]
    assert fresh["target_mask"][0].tolist() == [True, False]


def _actor(slots: int = 5) -> SkyArenaActor:
    return SkyArenaActor(
        self_dim=SELF_FEATURE_DIM,
        entity_dim=ENTITY_FEATURE_DIM,
        candidate_slots=slots,
        num_agents=3,
        trunk_dim=32,
        lstm_hidden_dim=32,
        entity_embed_dim=16,
        attention_heads=4,
    ).eval()


def test_entity_encoder_and_actor_are_permutation_equivariant() -> None:
    torch.manual_seed(4)
    batch_size, slots = 2, 5
    actor = _actor(slots)
    features = torch.randn(batch_size, slots, ENTITY_FEATURE_DIM)
    mask = torch.tensor(
        [[True, True, False, True, False], [True, False, True, True, True]]
    )
    batch = {
        "self_features": torch.randn(batch_size, SELF_FEATURE_DIM),
        "entity_features": features,
        "entity_mask": mask,
        "agent_id": torch.tensor([0, 1]),
    }
    hidden = (torch.zeros(batch_size, 32), torch.zeros(batch_size, 32))
    original = actor.step(batch, hidden)

    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted_batch = dict(batch)
    permuted_batch["entity_features"] = features[:, permutation]
    permuted_batch["entity_mask"] = mask[:, permutation]
    permuted = actor.step(permuted_batch, hidden)

    torch.testing.assert_close(original["course_logits"], permuted["course_logits"])
    torch.testing.assert_close(original["target_logits"][:, 0], permuted["target_logits"][:, 0])
    torch.testing.assert_close(
        original["target_logits"][:, 1:][:, permutation],
        permuted["target_logits"][:, 1:],
    )
    torch.testing.assert_close(
        original["fire_logits"][:, 1:][:, permutation],
        permuted["fire_logits"][:, 1:],
    )


def test_actor_step_and_sequence_shapes_are_finite_with_empty_entity_sets() -> None:
    actor = _actor(5)
    batch_size, seq_len, slots = 3, 4, 5
    step_batch = {
        "self_features": torch.zeros(batch_size, SELF_FEATURE_DIM),
        "entity_features": torch.zeros(batch_size, slots, ENTITY_FEATURE_DIM),
        "entity_mask": torch.zeros(batch_size, slots, dtype=torch.bool),
        "agent_id": torch.arange(batch_size),
    }
    hidden = (torch.zeros(batch_size, 32), torch.zeros(batch_size, 32))
    step_out = actor.step(step_batch, hidden)
    assert step_out["course_logits"].shape == (batch_size, 9)
    assert step_out["target_logits"].shape == (batch_size, slots + 1)
    assert step_out["fire_logits"].shape == (batch_size, slots + 1, 3)
    assert all(torch.isfinite(value).all() for value in step_out.values())

    sequence_batch = {
        key: value.unsqueeze(0).expand(seq_len, *value.shape)
        for key, value in step_batch.items()
    }
    sequence_out = actor.forward_sequence(
        sequence_batch,
        hidden,
        episode_starts=torch.zeros(seq_len, batch_size, dtype=torch.bool),
    )
    assert sequence_out["course_logits"].shape == (seq_len, batch_size, 9)
    assert sequence_out["target_logits"].shape == (seq_len, batch_size, slots + 1)
    assert sequence_out["fire_logits"].shape == (
        seq_len,
        batch_size,
        slots + 1,
        3,
    )


def test_standalone_set_encoder_handles_all_padding() -> None:
    encoder = EntityEncoder(ENTITY_FEATURE_DIM, 16, num_heads=4)
    tokens, pooled = encoder(
        torch.zeros(2, 5, ENTITY_FEATURE_DIM),
        torch.zeros(2, 5, dtype=torch.bool),
    )
    assert tokens.shape == (2, 5, 16)
    assert pooled.shape == (2, 16)
    assert torch.isfinite(tokens).all()
    assert torch.isfinite(pooled).all()
    assert torch.count_nonzero(tokens) == 0
    assert torch.count_nonzero(pooled) == 0
