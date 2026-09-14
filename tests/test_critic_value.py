"""Tests for the agent-conditioned critic and PopArt value head."""

from __future__ import annotations

import torch

from skyarena2d.rl.models.critic import PopArtValueHead, SkyArenaCritic


def test_agent_conditioned_critic_shapes_and_identity_conditioning() -> None:
    torch.manual_seed(11)
    critic = SkyArenaCritic(global_state_dim=7, hidden_dim=32, num_agents=4)
    global_state = torch.randn(3, 7)

    all_values = critic(global_state)
    selected_values = critic(global_state, torch.tensor([0, 1, 2]))
    subset_values = critic(global_state, torch.tensor([1, 3]))

    assert all_values.shape == (3, 4)
    assert selected_values.shape == (3,)
    assert subset_values.shape == (3, 2)
    torch.testing.assert_close(selected_values, all_values[torch.arange(3), torch.tensor([0, 1, 2])])
    torch.testing.assert_close(subset_values, all_values[:, [1, 3]])
    assert not torch.allclose(all_values[:, 0], all_values[:, 1])


def test_batched_rollout_values_match_stepwise_evaluation() -> None:
    torch.manual_seed(7)
    critic = SkyArenaCritic(global_state_dim=7, hidden_dim=32, num_agents=4)
    states = torch.randn(5, 3, 7)

    stepwise = torch.stack([critic(states[step]) for step in range(states.shape[0])])
    batched = critic(states.reshape(-1, 7)).reshape(5, 3, 4)

    torch.testing.assert_close(batched, stepwise, rtol=1e-6, atol=1e-6)


def test_popart_updates_stats_while_preserving_denormalized_output() -> None:
    torch.manual_seed(23)
    head = PopArtValueHead(5, beta=0.5)
    features = torch.randn(12, 5)
    before = head.denormalize(head(features)).detach().clone()

    targets = torch.linspace(-20.0, 40.0, 31)
    head.update(targets)
    after = head.denormalize(head(features)).detach()

    assert head.running_count.item() == targets.numel()
    assert head.running_mean.item() == torch.mean(targets).item()
    assert head.running_std.item() > 1.0
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)

    before_second_update = after.clone()
    second_targets = torch.linspace(50.0, 80.0, 17)
    expected_mean = 0.5 * targets.mean() + 0.5 * second_targets.mean()
    head.update(second_targets)
    after_second_update = head.denormalize(head(features)).detach()

    assert head.running_count.item() == targets.numel() + second_targets.numel()
    torch.testing.assert_close(head.running_mean, expected_mean)
    torch.testing.assert_close(after_second_update, before_second_update, rtol=1e-5, atol=1e-5)


def test_popart_critic_exposes_normalized_training_values() -> None:
    torch.manual_seed(29)
    critic = SkyArenaCritic(
        global_state_dim=6,
        hidden_dim=24,
        num_agents=3,
        use_popart=True,
        popart_beta=0.5,
    )
    global_state = torch.randn(4, 6)
    before = critic(global_state).detach().clone()
    targets = torch.randn(20, 3) * 8.0 + 13.0

    critic.update_popart(targets)
    after = critic(global_state).detach()
    normalized = critic(global_state, normalized=True)

    assert before.shape == (4, 3)
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(critic.value_head.denormalize(normalized), after)
    torch.testing.assert_close(critic.normalize_targets(targets).mean(), torch.zeros(()), atol=1e-5, rtol=0.0)
