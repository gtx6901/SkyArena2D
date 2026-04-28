from __future__ import annotations

import numpy as np
import torch

from skyarena2d.rl.algo.rollout import (
    INTERCEPT_TARGET,
    NEAREST_VISIBLE,
    SELECTED_TARGET,
    build_batched_movement_mode_mask,
    sample_policy_actions,
)


class _DummyActor:
    def step(self, batch, hidden_state):
        bsz = batch["self_features"].shape[0]
        device = batch["self_features"].device

        movement_mode_logits = torch.zeros((bsz, 9), dtype=torch.float32, device=device)
        course_logits = torch.zeros((bsz, 32), dtype=torch.float32, device=device)
        search_goal_logits = torch.zeros((bsz, 64), dtype=torch.float32, device=device)
        target_logits = torch.full((bsz, 7), -10.0, dtype=torch.float32, device=device)
        target_logits[:, 1] = 10.0

        fire_logits = torch.full((bsz, 7, 3), -10.0, dtype=torch.float32, device=device)
        fire_logits[:, :, 0] = 0.0
        fire_logits[:, 1, 1] = 10.0

        h, c = hidden_state
        return {
            "movement_mode_logits": movement_mode_logits,
            "course_logits": course_logits,
            "search_goal_logits": search_goal_logits,
            "target_logits": target_logits,
            "fire_logits": fire_logits,
            "next_h": h,
            "next_c": c,
        }


def _build_obs(entity_mask_value: bool) -> dict[str, np.ndarray]:
    obs = {
        "self_features": np.zeros((1, 1, 20), dtype=np.float32),
        "entity_features": np.zeros((1, 1, 6, 10), dtype=np.float32),
        "entity_mask": np.zeros((1, 1, 6), dtype=bool),
        "semantic_map": np.zeros((1, 1, 9, 16, 16), dtype=np.float32),
        "current_search_goal_id": np.zeros((1, 1), dtype=np.int64),
        "agent_id": np.zeros((1, 1), dtype=np.int64),
        "region_features": np.zeros((1, 1, 64, 10), dtype=np.float32),
        "course_mask": np.ones((1, 1, 32), dtype=bool),
        "search_goal_mask": np.ones((1, 1, 64), dtype=bool),
        "target_mask": np.zeros((1, 1, 7), dtype=bool),
        "alive_mask": np.ones((1, 1), dtype=np.float32),
        "has_active_contact": np.ones((1, 1), dtype=np.float32),
        "candidate_ids": np.full((1, 1, 6), -1, dtype=np.int64),
        "candidate_can_long": np.zeros((1, 1, 6), dtype=bool),
        "candidate_can_short": np.zeros((1, 1, 6), dtype=bool),
        "global_state": np.zeros((1, 181), dtype=np.float32),
    }
    obs["entity_mask"][0, 0, 0] = entity_mask_value
    obs["target_mask"][0, 0, 0] = True
    obs["target_mask"][0, 0, 1] = True
    obs["candidate_ids"][0, 0, 0] = 0
    obs["candidate_can_long"][0, 0, 0] = True
    return obs


def test_enemy_zero_candidate_is_valid_target_opportunity() -> None:
    actor = _DummyActor()
    h = torch.zeros((1, 1, 8), dtype=torch.float32)
    c = torch.zeros((1, 1, 8), dtype=torch.float32)

    out = sample_policy_actions(
        actor,
        _build_obs(entity_mask_value=True),
        (h, c),
        torch.device("cpu"),
        deterministic=True,
    )

    assert int(out["target"][0, 0]) == 1
    assert int(out["fire"][0, 0]) == 1


def test_target_opportunity_uses_entity_mask() -> None:
    actor = _DummyActor()
    h = torch.zeros((1, 1, 8), dtype=torch.float32)
    c = torch.zeros((1, 1, 8), dtype=torch.float32)

    out = sample_policy_actions(
        actor,
        _build_obs(entity_mask_value=False),
        (h, c),
        torch.device("cpu"),
        deterministic=True,
    )

    assert int(out["target"][0, 0]) == 0
    assert int(out["fire"][0, 0]) == 0


def test_movement_mode_mask_requires_target_for_selected_modes() -> None:
    target = torch.tensor([[0, 1]])
    alive = torch.tensor([[True, True]])
    entity = torch.tensor([[[True, False], [True, False]]])
    contact = torch.tensor([[True, True]])

    mask = build_batched_movement_mode_mask(
        target_action=target,
        alive_mask=alive,
        entity_mask=entity,
        has_active_contact=contact,
    )

    assert mask[0, 0, NEAREST_VISIBLE]
    assert not mask[0, 0, SELECTED_TARGET]
    assert not mask[0, 0, INTERCEPT_TARGET]
    assert mask[0, 1, SELECTED_TARGET]


def test_movement_mode_mask_blocks_nearest_without_entity() -> None:
    target = torch.tensor([[0]])
    alive = torch.tensor([[True]])
    entity = torch.zeros((1, 1, 2), dtype=torch.bool)
    contact = torch.tensor([[False]])

    mask = build_batched_movement_mode_mask(
        target_action=target,
        alive_mask=alive,
        entity_mask=entity,
        has_active_contact=contact,
    )

    assert not mask[0, 0, NEAREST_VISIBLE]
