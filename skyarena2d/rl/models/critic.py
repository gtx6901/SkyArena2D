"""SkyArena MAPPO centralized critic."""
from __future__ import annotations

import torch
from torch import nn

from .encoders import mlp


class SkyArenaCritic(nn.Module):
    """Simple centralized MLP critic over the team global state."""

    def __init__(self, global_state_dim: int, hidden_dim: int):
        super().__init__()
        self.value_net = mlp(global_state_dim, [hidden_dim, hidden_dim], 1)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.value_net(global_state).squeeze(-1)
