"""Neural-network encoder building blocks for SkyArena MAPPO."""
from __future__ import annotations

import torch
from torch import nn


def mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Build a small feed-forward network."""
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(last_dim, hidden_dim), activation()))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class _SetAttentionBlock(nn.Module):
    """Permutation-equivariant masked self-attention block."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, tokens: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = self.attention_norm(tokens + attended)
        return self.output_norm(tokens + self.feed_forward(tokens))


class EntityEncoder(nn.Module):
    """Encode a padded entity set and retain every token for pointer heads.

    No slot-position embedding is used: permuting entity tokens and their mask
    permutes the returned slot embeddings while leaving the pooled embedding
    unchanged (up to floating-point reduction noise).
    """

    def __init__(
        self,
        entity_dim: int,
        embed_dim: int,
        *,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"entity embed_dim={embed_dim} must be divisible by heads={num_heads}"
            )
        self.input_encoder = nn.Sequential(
            nn.Linear(entity_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            _SetAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)
        )
        self.pool_score = nn.Linear(embed_dim, 1)

    def forward(
        self, entity_features: torch.Tensor, entity_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if entity_features.ndim != 3:
            raise ValueError("entity_features must have shape [batch, slots, dim]")
        mask = entity_mask.to(dtype=torch.bool)
        if mask.shape != entity_features.shape[:2]:
            raise ValueError("entity_mask must match entity_features [batch, slots]")

        tokens = self.input_encoder(entity_features)
        # MultiheadAttention cannot consume a row whose keys are all masked.
        safe_mask = mask.clone()
        has_entity = safe_mask.any(dim=1)
        if safe_mask.shape[1] and torch.any(~has_entity):
            safe_mask[~has_entity, 0] = True
        key_padding_mask = ~safe_mask
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask)
            tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)

        scores = self.pool_score(tokens).squeeze(-1)
        scores = scores.masked_fill(~safe_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return tokens, pooled
