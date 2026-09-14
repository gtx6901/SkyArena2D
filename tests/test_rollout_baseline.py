import math

import numpy as np
import torch

from skyarena2d.rl.algo.rollout import (
    compute_gae,
    evaluate_policy_heads,
    sample_policy_actions,
)


class _DeterministicActor:
    def step(self, obs, hidden):
        batch_size, entity_slots = obs["entity_mask"].shape
        device = obs["entity_mask"].device
        course_logits = torch.zeros((batch_size, 9), device=device)
        course_logits[:, 4] = 5.0
        target_logits = torch.zeros((batch_size, entity_slots + 1), device=device)
        target_logits[:, 1] = 100.0  # masked ally/invalid target
        target_logits[:, 3] = 10.0
        fire_logits = torch.zeros((batch_size, entity_slots + 1, 3), device=device)
        fire_logits[:, 3, 1] = 100.0  # masked long-range shot
        fire_logits[:, 3, 2] = 10.0
        return {
            "course_logits": course_logits,
            "target_logits": target_logits,
            "fire_logits": fire_logits,
            "next_h": hidden[0] + 1,
            "next_c": hidden[1] + 2,
        }


def test_sampling_uses_pointer_and_selected_target_weapon_masks() -> None:
    obs = {
        "self_features": np.zeros((1, 2, 4), dtype=np.float32),
        "entity_features": np.zeros((1, 2, 3, 5), dtype=np.float32),
        "entity_mask": np.array([[[True, False, True], [True, False, True]]]),
        "target_mask": np.array([[[True, False, False, True], [True, False, False, True]]]),
        "candidate_can_long": np.zeros((1, 2, 3), dtype=bool),
        "candidate_can_short": np.array([[[False, False, True], [False, False, True]]]),
        "alive_mask": np.array([[1.0, 0.0]], dtype=np.float32),
        "agent_id": np.array([[0, 1]], dtype=np.int64),
    }
    hidden = (torch.zeros((1, 2, 6)), torch.zeros((1, 2, 6)))
    result = sample_policy_actions(
        _DeterministicActor(), obs, hidden, torch.device("cpu"), True,
        return_diagnostics=True,
    )

    assert result["course"].tolist() == [[4, 0]]
    assert result["target"].tolist() == [[3, 0]]
    assert result["fire"].tolist() == [[2, 0]]
    assert result["fire_mask"].tolist() == [[[True, False, True], [True, False, False]]]
    assert result["log_prob"][0, 1] == 0.0
    assert result["next_h"].shape == (1, 2, 6)


def test_fast_sampling_log_prob_matches_full_head_evaluation() -> None:
    obs = {
        "self_features": np.zeros((1, 2, 4), dtype=np.float32),
        "entity_features": np.zeros((1, 2, 3, 5), dtype=np.float32),
        "entity_mask": np.ones((1, 2, 3), dtype=bool),
        "target_mask": np.ones((1, 2, 4), dtype=bool),
        "candidate_can_long": np.ones((1, 2, 3), dtype=bool),
        "candidate_can_short": np.ones((1, 2, 3), dtype=bool),
        "alive_mask": np.ones((1, 2), dtype=np.float32),
        "agent_id": np.array([[0, 1]], dtype=np.int64),
        "global_state": np.full((1, 7), np.nan, dtype=np.float32),
    }
    hidden = (torch.zeros((1, 2, 6)), torch.zeros((1, 2, 6)))
    fast = sample_policy_actions(
        _DeterministicActor(), obs, hidden, torch.device("cpu"), True
    )
    diagnostic = sample_policy_actions(
        _DeterministicActor(), obs, hidden, torch.device("cpu"), True,
        return_diagnostics=True,
    )

    np.testing.assert_array_equal(fast["course"], diagnostic["course"])
    np.testing.assert_array_equal(fast["target"], diagnostic["target"])
    np.testing.assert_array_equal(fast["fire"], diagnostic["fire"])
    np.testing.assert_allclose(fast["log_prob"], diagnostic["log_prob"], atol=1e-7)


def test_all_three_policy_heads_contribute_entropy_and_log_prob() -> None:
    evaluated = evaluate_policy_heads(
        course_logits=torch.zeros((1, 9)),
        target_logits=torch.zeros((1, 3)),
        fire_logits=torch.zeros((1, 3, 3)),
        course_action=torch.tensor([0]),
        target_action=torch.tensor([2]),
        fire_action=torch.tensor([1]),
        entity_mask=torch.tensor([[True, True]]),
        target_mask=torch.tensor([[True, False, True]]),
        alive_mask=torch.tensor([True]),
        candidate_can_long=torch.tensor([[False, True]]),
        candidate_can_short=torch.tensor([[False, True]]),
    )
    expected_entropy = math.log(9) + math.log(2) + math.log(3)
    assert torch.allclose(evaluated["entropy"], torch.tensor([expected_entropy]))
    assert torch.allclose(evaluated["log_prob"], -torch.tensor([expected_entropy]))
    assert torch.all(evaluated["head_entropy"] > 0)


def test_gae_is_per_agent_and_broadcasts_environment_done() -> None:
    reward = np.array([
        [[1.0, 10.0], [2.0, 20.0]],
        [[3.0, 30.0], [4.0, 40.0]],
    ], dtype=np.float32)
    value = np.zeros_like(reward)
    done = np.array([[False, True], [True, True]])
    advantages, returns = compute_gae(
        reward, value, np.zeros((2, 2), dtype=np.float32), done, gamma=1.0, gae_lambda=1.0,
    )
    np.testing.assert_allclose(advantages[0, 0], [4.0, 40.0])
    np.testing.assert_allclose(advantages[0, 1], [2.0, 20.0])
    np.testing.assert_allclose(returns, advantages)
