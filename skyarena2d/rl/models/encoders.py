"""Neural network encoder building blocks for SkyArena MAPPO."""
from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation=nn.ReLU) -> nn.Sequential:
    """Build a simple MLP."""
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class EntityEncoder(nn.Module):
    """Encode fixed-slot enemy candidates and pool them for the actor trunk."""

    def __init__(self, entity_dim: int, embed_dim: int):
        super().__init__()
        self.slot_encoder = mlp(entity_dim, [embed_dim], embed_dim)

    def forward(self, entity_features: torch.Tensor, entity_mask: torch.Tensor):
        # entity_features: [B, K, D]
        slot_embeddings = self.slot_encoder(entity_features)
        mask = entity_mask.unsqueeze(-1).float()
        masked_embeddings = slot_embeddings * mask
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        pooled = masked_embeddings.sum(dim=1) / denom
        return slot_embeddings, pooled


class SemanticMapEncoder(nn.Module):
    """Small CNN for the semantic screen branch."""

    def __init__(self, in_channels: int, output_dim: int, input_size: int, pooling: str = "spatial_flatten"):
        super().__init__()
        if str(pooling).strip().lower() != "spatial_flatten":
            raise ValueError(f"Unsupported map encoder pooling: {pooling}")
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            conv_out = self.conv(dummy)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(conv_out.numel()), output_dim),
            nn.ReLU(),
        )

    def forward(self, semantic_map: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(semantic_map))
