"""Compact entity-set recurrent actor for SkyArena MAPPO."""
from __future__ import annotations

import torch
from torch import nn

from .encoders import EntityEncoder, mlp


class SkyArenaActor(nn.Module):
    """Recurrent actor over agent-centric entity tokens.

    The policy exposes three atomic decisions: a relative course change, an
    attention/engagement target, and a weapon choice conditioned on each target
    token.  ``fire_logits[:, target_action]`` is therefore the categorical fire
    distribution for the selected target.
    """

    def __init__(
        self,
        *,
        self_dim: int,
        entity_dim: int,
        candidate_slots: int,
        num_agents: int,
        trunk_dim: int,
        lstm_hidden_dim: int,
        entity_embed_dim: int,
        course_bins: int = 9,
        agent_id_embed_dim: int = 16,
        attention_heads: int = 4,
        attention_layers: int = 2,
    ) -> None:
        super().__init__()
        if int(course_bins) != 9:
            raise ValueError("compact baseline requires exactly 9 course actions")
        self.course_bins = 9
        self.candidate_slots = int(candidate_slots)
        self.num_agents = int(num_agents)
        self.fire_action_dim = 3
        self.lstm_hidden_dim = int(lstm_hidden_dim)

        self.self_encoder = mlp(self_dim, [trunk_dim], trunk_dim)
        self.entity_encoder = EntityEncoder(
            entity_dim=entity_dim,
            embed_dim=entity_embed_dim,
            num_heads=attention_heads,
            num_layers=attention_layers,
        )
        self.agent_embedding = nn.Embedding(
            self.num_agents + 1, int(agent_id_embed_dim)
        )
        fusion_dim = trunk_dim + entity_embed_dim + int(agent_id_embed_dim)
        self.fusion_trunk = mlp(fusion_dim, [trunk_dim], lstm_hidden_dim)
        self.lstm = nn.LSTMCell(lstm_hidden_dim, lstm_hidden_dim)

        self.course_head = nn.Linear(lstm_hidden_dim, self.course_bins)
        self.null_target_embedding = nn.Parameter(torch.zeros(entity_embed_dim))
        joint_dim = lstm_hidden_dim + entity_embed_dim
        self.target_scorer = nn.Sequential(
            nn.Linear(joint_dim, lstm_hidden_dim),
            nn.Tanh(),
            nn.Linear(lstm_hidden_dim, 1),
        )
        self.fire_scorer = nn.Sequential(
            nn.Linear(joint_dim, lstm_hidden_dim),
            nn.Tanh(),
            nn.Linear(lstm_hidden_dim, self.fire_action_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Use PPO-friendly orthogonal initialization throughout the actor."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("tanh"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.orthogonal_(module.weight)

        for weight_name in ("weight_ih", "weight_hh"):
            weight = getattr(self.lstm, weight_name)
            for gate_weight in weight.chunk(4, dim=0):
                nn.init.orthogonal_(gate_weight)
        nn.init.zeros_(self.lstm.bias_ih)
        nn.init.zeros_(self.lstm.bias_hh)
        # A positive forget bias is a stable recurrent-policy default.
        hidden = self.lstm_hidden_dim
        with torch.no_grad():
            self.lstm.bias_ih[hidden : 2 * hidden].fill_(1.0)

        nn.init.orthogonal_(self.course_head.weight, gain=0.01)
        nn.init.zeros_(self.course_head.bias)
        nn.init.orthogonal_(self.target_scorer[-1].weight, gain=0.01)
        nn.init.zeros_(self.target_scorer[-1].bias)
        nn.init.orthogonal_(self.fire_scorer[-1].weight, gain=0.01)
        nn.init.zeros_(self.fire_scorer[-1].bias)

    def _encode_inputs(
        self,
        self_features: torch.Tensor,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
        agent_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self_embed = self.self_encoder(self_features)
        slot_embed, pooled_entity = self.entity_encoder(
            entity_features, entity_mask
        )
        agent_embed = self.agent_embedding(
            torch.clamp(agent_id.long(), min=0, max=self.num_agents)
        )
        fused = self.fusion_trunk(
            torch.cat((self_embed, pooled_entity, agent_embed), dim=-1)
        )
        return fused, slot_embed

    def step(
        self,
        batch: dict[str, torch.Tensor],
        hidden_state: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        fused, slot_embed = self._encode_inputs(
            self_features=batch["self_features"],
            entity_features=batch["entity_features"],
            entity_mask=batch["entity_mask"],
            agent_id=batch["agent_id"],
        )
        h, c = self.lstm(fused, hidden_state)
        outputs = self._project_heads(h, slot_embed)
        outputs["next_h"] = h
        outputs["next_c"] = c
        return outputs

    def forward_sequence(
        self,
        batch: dict[str, torch.Tensor],
        hidden_state: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        seq_len, batch_size = batch["self_features"].shape[:2]
        fused, slot_embed = self._encode_inputs(
            self_features=batch["self_features"].reshape(
                seq_len * batch_size, -1
            ),
            entity_features=batch["entity_features"].reshape(
                seq_len * batch_size, *batch["entity_features"].shape[2:]
            ),
            entity_mask=batch["entity_mask"].reshape(seq_len * batch_size, -1),
            agent_id=batch["agent_id"].reshape(seq_len * batch_size),
        )
        fused = fused.reshape(seq_len, batch_size, -1)
        slot_embed = slot_embed.reshape(
            seq_len,
            batch_size,
            slot_embed.shape[1],
            slot_embed.shape[2],
        )

        h, c = hidden_state
        course_logits: list[torch.Tensor] = []
        target_logits: list[torch.Tensor] = []
        fire_logits: list[torch.Tensor] = []
        for step_idx in range(seq_len):
            if episode_starts is not None:
                start_mask = episode_starts[step_idx].to(h.dtype).unsqueeze(-1)
                h = h * (1.0 - start_mask)
                c = c * (1.0 - start_mask)
            h, c = self.lstm(fused[step_idx], (h, c))
            outputs = self._project_heads(h, slot_embed[step_idx])
            course_logits.append(outputs["course_logits"])
            target_logits.append(outputs["target_logits"])
            fire_logits.append(outputs["fire_logits"])

        return {
            "course_logits": torch.stack(course_logits, dim=0),
            "target_logits": torch.stack(target_logits, dim=0),
            "fire_logits": torch.stack(fire_logits, dim=0),
            "final_h": h,
            "final_c": c,
        }

    def _project_heads(
        self, hidden: torch.Tensor, slot_embed: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch_size = hidden.shape[0]
        null_embed = self.null_target_embedding.view(1, 1, -1).expand(
            batch_size, -1, -1
        )
        target_tokens = torch.cat((null_embed, slot_embed), dim=1)
        repeated_hidden = hidden.unsqueeze(1).expand(
            -1, target_tokens.shape[1], -1
        )
        joint = torch.cat((repeated_hidden, target_tokens), dim=-1)
        return {
            "course_logits": self.course_head(hidden),
            "target_logits": self.target_scorer(joint).squeeze(-1),
            "fire_logits": self.fire_scorer(joint),
        }
