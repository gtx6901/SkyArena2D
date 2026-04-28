"""SkyArena MAPPO actor network."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from .encoders import EntityEncoder, SemanticMapEncoder, mlp


class SkyArenaActor(nn.Module):
    """Entity + semantic-map recurrent actor with persistent search-goal conditioning."""

    def __init__(
        self,
        *,
        self_dim: int,
        entity_dim: int,
        map_channels: int,
        candidate_slots: int,
        num_agents: int,
        course_bins: int,
        search_goal_bins: int,
        region_feature_dim: int,
        trunk_dim: int,
        lstm_hidden_dim: int,
        entity_embed_dim: int,
        map_embed_dim: int,
        semantic_map_size: int,
        current_goal_embed_dim: int = 32,
        agent_id_embed_dim: int = 16,
        region_embed_dim: int = 64,
        map_encoder_pooling: str = "spatial_flatten",
    ):
        super().__init__()
        self.course_bins = int(course_bins)
        self.search_goal_bins = int(search_goal_bins)
        self.candidate_slots = int(candidate_slots)
        self.num_agents = int(num_agents)
        self.movement_mode_bins = 9
        self.fire_action_dim = 3
        self.lstm_hidden_dim = int(lstm_hidden_dim)

        self.self_encoder = mlp(self_dim, [trunk_dim], trunk_dim)
        self.entity_encoder = EntityEncoder(entity_dim=entity_dim, embed_dim=entity_embed_dim)
        self.map_encoder = SemanticMapEncoder(
            in_channels=map_channels,
            output_dim=map_embed_dim,
            input_size=semantic_map_size,
            pooling=map_encoder_pooling,
        )
        self.goal_embedding = nn.Embedding(self.search_goal_bins + 1, int(current_goal_embed_dim))
        self.agent_embedding = nn.Embedding(self.num_agents + 1, int(agent_id_embed_dim))
        self.region_encoder = mlp(region_feature_dim, [region_embed_dim], region_embed_dim)
        self.region_query = nn.Linear(lstm_hidden_dim, region_embed_dim)
        self.region_bias = nn.Linear(region_embed_dim, 1)
        fusion_input_dim = trunk_dim + entity_embed_dim + map_embed_dim + int(current_goal_embed_dim) + int(agent_id_embed_dim)
        self.fusion_trunk = mlp(fusion_input_dim, [trunk_dim], lstm_hidden_dim)
        self.lstm = nn.LSTMCell(lstm_hidden_dim, lstm_hidden_dim)

        self.movement_mode_head = nn.Linear(lstm_hidden_dim, self.movement_mode_bins)
        self.course_head = nn.Linear(lstm_hidden_dim, self.course_bins)
        self.null_target_embedding = nn.Parameter(torch.zeros(entity_embed_dim))
        self.target_scorer = nn.Sequential(
            nn.Linear(lstm_hidden_dim + entity_embed_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim, 1),
        )
        self.fire_scorer = nn.Sequential(
            nn.Linear(lstm_hidden_dim + entity_embed_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim, self.fire_action_dim),
        )

    def _encode_inputs(
        self,
        self_features: torch.Tensor,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
        semantic_map: torch.Tensor,
        current_search_goal_id: torch.Tensor,
        agent_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self_embed = self.self_encoder(self_features)
        slot_embed, pooled_entity = self.entity_encoder(entity_features, entity_mask)
        map_embed = self.map_encoder(semantic_map)
        goal_embed = self.goal_embedding(torch.clamp(current_search_goal_id.long(), min=0, max=self.search_goal_bins))
        agent_embed = self.agent_embedding(torch.clamp(agent_id.long(), min=0, max=self.num_agents))
        fused = self.fusion_trunk(torch.cat([self_embed, pooled_entity, map_embed, goal_embed, agent_embed], dim=-1))
        return fused, slot_embed

    def step(
        self,
        batch: Dict[str, torch.Tensor],
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        fused, slot_embed = self._encode_inputs(
            self_features=batch["self_features"],
            entity_features=batch["entity_features"],
            entity_mask=batch["entity_mask"],
            semantic_map=batch["semantic_map"],
            current_search_goal_id=batch["current_search_goal_id"],
            agent_id=batch["agent_id"],
        )
        h, c = self.lstm(fused, hidden_state)
        outputs = self._project_heads(h, slot_embed, batch["region_features"])
        outputs["next_h"] = h
        outputs["next_c"] = c
        return outputs

    def forward_sequence(
        self,
        batch: Dict[str, torch.Tensor],
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        seq_len, batch_size = batch["self_features"].shape[:2]
        fused, slot_embed = self._encode_inputs(
            self_features=batch["self_features"].reshape(seq_len * batch_size, -1),
            entity_features=batch["entity_features"].reshape(seq_len * batch_size, *batch["entity_features"].shape[2:]),
            entity_mask=batch["entity_mask"].reshape(seq_len * batch_size, -1),
            semantic_map=batch["semantic_map"].reshape(seq_len * batch_size, *batch["semantic_map"].shape[2:]),
            current_search_goal_id=batch["current_search_goal_id"].reshape(seq_len * batch_size),
            agent_id=batch["agent_id"].reshape(seq_len * batch_size),
        )
        fused = fused.reshape(seq_len, batch_size, -1)
        slot_embed = slot_embed.reshape(seq_len, batch_size, slot_embed.shape[1], slot_embed.shape[2])
        region_features = batch["region_features"].reshape(
            seq_len, batch_size, batch["region_features"].shape[-2], batch["region_features"].shape[-1],
        )

        h, c = hidden_state
        movement_mode_logits, course_logits, search_goal_logits, target_logits, fire_logits = [], [], [], [], []
        for step_idx in range(seq_len):
            if episode_starts is not None:
                start_mask = episode_starts[step_idx].float().unsqueeze(-1)
                h = h * (1.0 - start_mask)
                c = c * (1.0 - start_mask)
            h, c = self.lstm(fused[step_idx], (h, c))
            out = self._project_heads(h, slot_embed[step_idx], region_features[step_idx])
            movement_mode_logits.append(out["movement_mode_logits"])
            course_logits.append(out["course_logits"])
            search_goal_logits.append(out["search_goal_logits"])
            target_logits.append(out["target_logits"])
            fire_logits.append(out["fire_logits"])
        return {
            "movement_mode_logits": torch.stack(movement_mode_logits, dim=0),
            "course_logits": torch.stack(course_logits, dim=0),
            "search_goal_logits": torch.stack(search_goal_logits, dim=0),
            "target_logits": torch.stack(target_logits, dim=0),
            "fire_logits": torch.stack(fire_logits, dim=0),
            "final_h": h,
            "final_c": c,
        }

    def _project_heads(
        self,
        hidden: torch.Tensor,
        slot_embed: torch.Tensor,
        region_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = hidden.shape[0]
        region_embed = self.region_encoder(region_features.reshape(-1, region_features.shape[-1])).reshape(
            batch_size, region_features.shape[-2], -1,
        )
        region_query = self.region_query(hidden).unsqueeze(1)
        search_goal_logits = torch.sum(region_embed * region_query, dim=-1) + self.region_bias(region_embed).squeeze(-1)
        null_embed = self.null_target_embedding.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        target_slots = torch.cat([null_embed, slot_embed], dim=1)
        repeated_hidden = hidden.unsqueeze(1).expand(-1, target_slots.shape[1], -1)
        joint = torch.cat([repeated_hidden, target_slots], dim=-1)
        return {
            "movement_mode_logits": self.movement_mode_head(hidden),
            "course_logits": self.course_head(hidden),
            "search_goal_logits": search_goal_logits,
            "target_logits": self.target_scorer(joint).squeeze(-1),
            "fire_logits": self.fire_scorer(joint),
        }
