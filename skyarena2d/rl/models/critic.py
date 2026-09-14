"""Agent-conditioned centralized value function for SkyArena MAPPO."""
from __future__ import annotations

import math

import torch
from torch import nn


def _orthogonal_linear(layer: nn.Linear, gain: float) -> nn.Linear:
    """Initialize a linear layer with the PPO-friendly orthogonal scheme."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class PopArtValueHead(nn.Module):
    """Scalar PopArt head with output-preserving running-stat updates.

    The linear layer predicts normalized values. ``forward`` returns those raw
    normalized predictions; callers can use :meth:`denormalize` for environment
    value units. Updating the running statistics rescales the final layer so its
    denormalized predictions do not jump when the normalization changes.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        beta: float = 0.999,
        min_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"PopArt beta must be in [0, 1), got {beta}")
        if min_std <= 0.0:
            raise ValueError(f"PopArt min_std must be positive, got {min_std}")

        self.beta = float(beta)
        self.min_std = float(min_std)
        self.linear = _orthogonal_linear(nn.Linear(input_dim, 1), gain=1.0)
        self.register_buffer("running_mean", torch.zeros((), dtype=torch.float32))
        self.register_buffer("running_var", torch.ones((), dtype=torch.float32))
        self.register_buffer("running_count", torch.zeros((), dtype=torch.float64))

    @property
    def running_std(self) -> torch.Tensor:
        return self.running_var.clamp_min(self.min_std**2).sqrt()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.running_mean.to(device=values.device, dtype=values.dtype)
        std = self.running_std.to(device=values.device, dtype=values.dtype)
        return (values - mean) / std

    def denormalize(self, normalized_values: torch.Tensor) -> torch.Tensor:
        mean = self.running_mean.to(device=normalized_values.device, dtype=normalized_values.dtype)
        std = self.running_std.to(device=normalized_values.device, dtype=normalized_values.dtype)
        return normalized_values * std + mean

    @torch.no_grad()
    def update(self, targets: torch.Tensor) -> None:
        """Update normalization moments and preserve denormalized predictions."""
        samples = torch.as_tensor(
            targets,
            device=self.running_mean.device,
            dtype=self.running_mean.dtype,
        ).reshape(-1)
        samples = samples[torch.isfinite(samples)]
        if samples.numel() == 0:
            return

        old_mean = self.running_mean.clone()
        old_std = self.running_std.clone()
        batch_mean = samples.mean()
        batch_var = samples.var(unbiased=False)

        if self.running_count.item() == 0.0:
            new_mean = batch_mean
            new_var = batch_var
        else:
            # Update first and second moments. This includes the between-mean
            # term that a direct EMA of variances would omit.
            old_second = self.running_var + self.running_mean.square()
            batch_second = batch_var + batch_mean.square()
            new_mean = self.beta * self.running_mean + (1.0 - self.beta) * batch_mean
            new_second = self.beta * old_second + (1.0 - self.beta) * batch_second
            new_var = new_second - new_mean.square()

        new_var = new_var.clamp_min(self.min_std**2)
        new_std = new_var.sqrt()

        scale = old_std / new_std
        self.linear.weight.mul_(scale)
        self.linear.bias.copy_((old_std * self.linear.bias + old_mean - new_mean) / new_std)
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.running_count.add_(float(samples.numel()))


class SkyArenaCritic(nn.Module):
    """Shared agent-conditioned centralized critic.

    With ``agent_ids=None``, the critic evaluates every configured agent and
    returns ``[batch, num_agents]``. The historical one-agent construction
    remains source-compatible and returns ``[batch]``. Passing agent ids with
    the same leading shape as ``global_state`` selects one agent per state;
    passing a shared 1-D id list evaluates that list for every state.

    PopArt is optional. The public forward result remains in environment value
    units by default, while ``normalized=True`` exposes normalized predictions
    for the value loss.
    """

    def __init__(
        self,
        global_state_dim: int,
        hidden_dim: int,
        num_agents: int = 1,
        *,
        agent_id_embed_dim: int = 16,
        use_popart: bool = False,
        popart_beta: float = 0.999,
        popart_min_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if global_state_dim <= 0 or hidden_dim <= 0:
            raise ValueError("global_state_dim and hidden_dim must be positive")
        if num_agents <= 0 or agent_id_embed_dim <= 0:
            raise ValueError("num_agents and agent_id_embed_dim must be positive")

        self.global_state_dim = int(global_state_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_agents = int(num_agents)
        self.use_popart = bool(use_popart)

        self.state_encoder = nn.Sequential(
            _orthogonal_linear(nn.Linear(global_state_dim, hidden_dim), gain=math.sqrt(2.0)),
            nn.Tanh(),
            _orthogonal_linear(nn.Linear(hidden_dim, hidden_dim), gain=math.sqrt(2.0)),
            nn.Tanh(),
        )
        self.agent_embedding = nn.Embedding(num_agents, agent_id_embed_dim)
        nn.init.orthogonal_(self.agent_embedding.weight, gain=1.0)
        self.conditioned_trunk = nn.Sequential(
            _orthogonal_linear(
                nn.Linear(hidden_dim + agent_id_embed_dim, hidden_dim),
                gain=math.sqrt(2.0),
            ),
            nn.Tanh(),
        )
        if self.use_popart:
            self.value_head: nn.Module = PopArtValueHead(
                hidden_dim,
                beta=popart_beta,
                min_std=popart_min_std,
            )
        else:
            self.value_head = _orthogonal_linear(nn.Linear(hidden_dim, 1), gain=1.0)

    def _conditioned_features(
        self,
        global_state: torch.Tensor,
        agent_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, bool]:
        if global_state.shape[-1] != self.global_state_dim:
            raise ValueError(
                f"Expected global_state[..., {self.global_state_dim}], got {tuple(global_state.shape)}"
            )
        state_features = self.state_encoder(global_state)
        leading_shape = state_features.shape[:-1]
        squeeze_agent_axis = False

        if agent_ids is None:
            ids = torch.arange(self.num_agents, device=global_state.device)
            ids = ids.expand(*leading_shape, self.num_agents)
            state_features = state_features.unsqueeze(-2).expand(*leading_shape, self.num_agents, self.hidden_dim)
            squeeze_agent_axis = self.num_agents == 1
        else:
            ids = torch.as_tensor(agent_ids, device=global_state.device, dtype=torch.long)
            if tuple(ids.shape) == tuple(leading_shape):
                # One explicit agent identity for each state.
                squeeze_agent_axis = False
            elif ids.ndim == 1:
                # A shared selection list for every state in the batch.
                ids = ids.expand(*leading_shape, ids.shape[0])
                state_features = state_features.unsqueeze(-2).expand(
                    *leading_shape, ids.shape[-1], self.hidden_dim
                )
            elif tuple(ids.shape[:-1]) == tuple(leading_shape):
                state_features = state_features.unsqueeze(-2).expand(
                    *leading_shape, ids.shape[-1], self.hidden_dim
                )
            else:
                raise ValueError(
                    "agent_ids must match global_state leading dimensions or append a selection dimension"
                )

        if torch.any((ids < 0) | (ids >= self.num_agents)):
            raise ValueError(f"agent_ids must be in [0, {self.num_agents})")
        conditioned = torch.cat((state_features, self.agent_embedding(ids)), dim=-1)
        return self.conditioned_trunk(conditioned), squeeze_agent_axis

    def forward(
        self,
        global_state: torch.Tensor,
        agent_ids: torch.Tensor | None = None,
        *,
        normalized: bool = False,
    ) -> torch.Tensor:
        features, squeeze_agent_axis = self._conditioned_features(global_state, agent_ids)
        if self.use_popart:
            assert isinstance(self.value_head, PopArtValueHead)
            values = self.value_head(features)
            if not normalized:
                values = self.value_head.denormalize(values)
        else:
            values = self.value_head(features).squeeze(-1)
        if squeeze_agent_axis:
            values = values.squeeze(-1)
        return values

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize value targets when PopArt is enabled."""
        if not self.use_popart:
            return targets
        assert isinstance(self.value_head, PopArtValueHead)
        return self.value_head.normalize(targets)

    @torch.no_grad()
    def update_popart(self, targets: torch.Tensor) -> None:
        """Update PopArt moments. This is a no-op only when explicitly disabled."""
        if not self.use_popart:
            raise RuntimeError("update_popart requires use_popart=True")
        assert isinstance(self.value_head, PopArtValueHead)
        self.value_head.update(targets)
