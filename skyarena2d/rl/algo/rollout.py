"""Rollout data structures and sampling for the entity MAPPO baseline.

The policy exposes three decisions only: a nine-bin relative course, an entity
pointer (including ``no target``), and a weapon decision conditioned on that
pointer.  Sampling and PPO evaluation share the mask semantics in this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

_ACTOR_OBS_KEYS = (
    "self_features",
    "entity_features",
    "entity_mask",
    "target_mask",
    "candidate_can_long",
    "candidate_can_short",
    "alive_mask",
    "agent_id",
)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Build a categorical distribution after validating its boolean mask."""
    if logits.shape != mask.shape:
        raise ValueError(f"logits and mask shapes differ: {logits.shape} != {mask.shape}")
    mask = mask.bool()
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("every categorical row must contain at least one valid action")
    return Categorical(logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min))


def to_torch_batch(obs_batch: Mapping[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert an observation batch without maintaining a second schema here."""
    result: dict[str, torch.Tensor] = {}
    for key, value in obs_batch.items():
        array = np.asarray(value)
        if array.dtype == np.bool_:
            dtype = torch.bool
        elif np.issubdtype(array.dtype, np.integer):
            dtype = torch.long
        else:
            dtype = torch.float32
        result[key] = torch.as_tensor(array, dtype=dtype, device=device)
    return result


def allocate_batched_obs(example_obs: Mapping[str, np.ndarray], num_envs: int) -> dict[str, np.ndarray]:
    """Allocate the reusable environment batch used by the collector."""
    return {key: np.empty((num_envs, *value.shape), dtype=value.dtype) for key, value in example_obs.items()}


def allocate_rollout_obs(
    example_obs: Mapping[str, np.ndarray], rollout_steps: int, num_envs: int,
) -> dict[str, np.ndarray]:
    return {
        key: np.empty((rollout_steps, num_envs, *value.shape), dtype=value.dtype)
        for key, value in example_obs.items()
    }


def fill_batched_obs(
    dst_batch: dict[str, np.ndarray], obs_list: list[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Fill a preallocated batch, avoiding a second full observation stack."""
    if len(obs_list) != next(iter(dst_batch.values())).shape[0]:
        raise ValueError("obs_list length does not match the allocated environment batch")
    for env_idx, obs in enumerate(obs_list):
        for key, dst in dst_batch.items():
            dst[env_idx] = obs[key]
    return dst_batch


def _flatten_actor_obs(
    obs_t: Mapping[str, torch.Tensor], num_envs: int, num_agents: int,
) -> dict[str, torch.Tensor]:
    """Flatten only tensors carrying leading ``[environment, agent]`` axes."""
    return {
        key: value.reshape(num_envs * num_agents, *value.shape[2:])
        for key, value in obs_t.items()
        if value.ndim >= 2 and tuple(value.shape[:2]) == (num_envs, num_agents)
    }


def build_target_mask(
    *,
    target_logits: torch.Tensor,
    entity_mask: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    alive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a pointer mask whose zero index is always ``no target``.

    ``target_mask`` may either include the no-target slot or contain one entry
    per entity.  Dead agents are restricted to no-target.
    """
    pointer_size = target_logits.shape[-1]
    entity_mask = entity_mask.bool()
    if entity_mask.shape[-1] != pointer_size - 1:
        raise ValueError(
            f"entity mask has {entity_mask.shape[-1]} slots but target head expects {pointer_size - 1}"
        )
    if target_mask is None:
        selectable = entity_mask
    else:
        target_mask = target_mask.bool()
        if target_mask.shape[-1] == pointer_size:
            selectable = target_mask[..., 1:] & entity_mask
        elif target_mask.shape[-1] == pointer_size - 1:
            selectable = target_mask & entity_mask
        else:
            raise ValueError(
                f"target mask has {target_mask.shape[-1]} slots, expected {pointer_size - 1} or {pointer_size}"
            )
    mask = torch.cat((torch.ones_like(selectable[..., :1]), selectable), dim=-1)
    if alive_mask is not None:
        mask[..., 1:] &= alive_mask.bool().unsqueeze(-1)
    return mask


def build_fire_mask_from_selected_targets(
    *,
    target_action: torch.Tensor,
    alive_mask: torch.Tensor,
    candidate_can_long: torch.Tensor,
    candidate_can_short: torch.Tensor,
) -> torch.Tensor:
    """Mask weapons for the selected entity; index zero is always no-fire."""
    target_action = target_action.long()
    alive = alive_mask.bool()
    can_long = candidate_can_long.bool()
    can_short = candidate_can_short.bool()
    if can_long.shape != can_short.shape:
        raise ValueError("long- and short-range weapon masks must have identical shapes")
    if can_long.shape[:-1] != target_action.shape:
        raise ValueError("weapon masks must have shape target_action.shape + (entity_slots,)")

    entity_slots = can_long.shape[-1]
    in_range = (target_action > 0) & (target_action <= entity_slots)
    if entity_slots:
        slot = torch.clamp(target_action - 1, min=0, max=entity_slots - 1)
        selected_long = torch.gather(can_long, -1, slot.unsqueeze(-1)).squeeze(-1)
        selected_short = torch.gather(can_short, -1, slot.unsqueeze(-1)).squeeze(-1)
    else:
        selected_long = torch.zeros_like(in_range)
        selected_short = torch.zeros_like(in_range)

    mask = torch.zeros((*target_action.shape, 3), dtype=torch.bool, device=target_action.device)
    mask[..., 0] = True
    mask[..., 1] = alive & in_range & selected_long
    mask[..., 2] = alive & in_range & selected_short
    return mask


def select_target_conditioned_fire_logits(
    fire_logits: torch.Tensor, target_action: torch.Tensor,
) -> torch.Tensor:
    """Select weapon logits belonging to each sampled pointer target."""
    if fire_logits.ndim != target_action.ndim + 2 or fire_logits.shape[-1] != 3:
        raise ValueError("fire_logits must have shape target_action.shape + (pointer_slots, 3)")
    if fire_logits.shape[:-2] != target_action.shape:
        raise ValueError("fire_logits and target_action leading shapes differ")
    if torch.any((target_action < 0) | (target_action >= fire_logits.shape[-2])):
        raise ValueError("target action is outside the pointer head")
    index = target_action.long().unsqueeze(-1).unsqueeze(-1).expand(*target_action.shape, 1, 3)
    return torch.gather(fire_logits, -2, index).squeeze(-2)


def evaluate_policy_heads(
    *,
    course_logits: torch.Tensor,
    target_logits: torch.Tensor,
    fire_logits: torch.Tensor,
    course_action: torch.Tensor,
    target_action: torch.Tensor,
    fire_action: torch.Tensor,
    entity_mask: torch.Tensor,
    alive_mask: torch.Tensor,
    candidate_can_long: torch.Tensor,
    candidate_can_short: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate the exact three-head action distribution used by PPO."""
    alive = alive_mask.bool()
    if course_logits.shape[:-1] != alive.shape or course_logits.shape[-1] != 9:
        raise ValueError("course_logits must have shape alive_mask.shape + (9,)")

    course_mask = torch.ones_like(course_logits, dtype=torch.bool)
    course_mask = torch.where(alive.unsqueeze(-1), course_mask, torch.zeros_like(course_mask))
    course_mask[..., 0] |= ~alive
    pointer_mask = build_target_mask(
        target_logits=target_logits,
        entity_mask=entity_mask,
        target_mask=target_mask,
        alive_mask=alive,
    )
    weapon_mask = build_fire_mask_from_selected_targets(
        target_action=target_action,
        alive_mask=alive,
        candidate_can_long=candidate_can_long,
        candidate_can_short=candidate_can_short,
    )
    selected_fire_logits = select_target_conditioned_fire_logits(fire_logits, target_action)

    course_dist = masked_categorical(course_logits, course_mask)
    target_dist = masked_categorical(target_logits, pointer_mask)
    fire_dist = masked_categorical(selected_fire_logits, weapon_mask)
    head_log_prob = torch.stack(
        (
            course_dist.log_prob(course_action.long()),
            target_dist.log_prob(target_action.long()),
            fire_dist.log_prob(fire_action.long()),
        ),
        dim=-1,
    )
    head_entropy = torch.stack(
        (course_dist.entropy(), target_dist.entropy(), fire_dist.entropy()), dim=-1,
    )
    live = alive.to(course_logits.dtype)
    return {
        "log_prob": head_log_prob.sum(dim=-1) * live,
        "entropy": head_entropy.sum(dim=-1) * live,
        "head_log_prob": head_log_prob * live.unsqueeze(-1),
        "head_entropy": head_entropy * live.unsqueeze(-1),
        "course_mask": course_mask,
        "target_mask": pointer_mask,
        "fire_mask": weapon_mask,
        "selected_fire_logits": selected_fire_logits,
    }


@torch.no_grad()
def sample_policy_actions(
    actor,
    obs_batch: dict[str, np.ndarray],
    hidden_state: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    deterministic: bool,
    *,
    return_diagnostics: bool = False,
):
    """Sample atomic course, pointer target, and target-conditioned weapon."""
    obs_t = to_torch_batch(
        {key: obs_batch[key] for key in _ACTOR_OBS_KEYS if key in obs_batch},
        device,
    )
    if "self_features" not in obs_t:
        raise KeyError("self_features is required to infer environment and agent axes")
    num_envs, num_agents = obs_t["self_features"].shape[:2]
    batch_size = num_envs * num_agents
    flat_obs = _flatten_actor_obs(obs_t, num_envs, num_agents)

    h, c = hidden_state
    out = actor.step(flat_obs, (h.reshape(batch_size, -1), c.reshape(batch_size, -1)))
    alive = obs_t["alive_mask"].reshape(batch_size).bool()
    entity_mask = obs_t["entity_mask"].reshape(batch_size, -1).bool()
    raw_target_mask = obs_t.get("target_mask")
    flat_target_mask = None if raw_target_mask is None else raw_target_mask.reshape(batch_size, -1).bool()

    course_mask = torch.ones_like(out["course_logits"], dtype=torch.bool)
    course_mask = torch.where(alive.unsqueeze(-1), course_mask, torch.zeros_like(course_mask))
    course_mask[:, 0] |= ~alive
    pointer_mask = build_target_mask(
        target_logits=out["target_logits"], entity_mask=entity_mask,
        target_mask=flat_target_mask, alive_mask=alive,
    )
    course_dist = masked_categorical(out["course_logits"], course_mask)
    target_dist = masked_categorical(out["target_logits"], pointer_mask)
    if deterministic:
        course_action = torch.argmax(
            out["course_logits"].masked_fill(
                ~course_mask, torch.finfo(out["course_logits"].dtype).min
            ),
            dim=-1,
        )
        target_action = torch.argmax(
            out["target_logits"].masked_fill(
                ~pointer_mask, torch.finfo(out["target_logits"].dtype).min
            ),
            dim=-1,
        )
    else:
        course_action = course_dist.sample()
        target_action = target_dist.sample()

    candidate_can_long = obs_t["candidate_can_long"].reshape(batch_size, -1).bool()
    candidate_can_short = obs_t["candidate_can_short"].reshape(batch_size, -1).bool()
    fire_mask = build_fire_mask_from_selected_targets(
        target_action=target_action, alive_mask=alive,
        candidate_can_long=candidate_can_long, candidate_can_short=candidate_can_short,
    )
    selected_fire_logits = select_target_conditioned_fire_logits(out["fire_logits"], target_action)
    fire_dist = masked_categorical(selected_fire_logits, fire_mask)
    if deterministic:
        fire_action = torch.argmax(
            selected_fire_logits.masked_fill(
                ~fire_mask, torch.finfo(selected_fire_logits.dtype).min
            ),
            dim=-1,
        )
    else:
        fire_action = fire_dist.sample()
    live = alive.to(out["course_logits"].dtype)
    log_prob = torch.stack(
        (
            course_dist.log_prob(course_action),
            target_dist.log_prob(target_action),
            fire_dist.log_prob(fire_action),
        ),
        dim=-1,
    ).sum(dim=-1) * live

    transfer = torch.stack(
        (
            course_action.to(torch.float32),
            target_action.to(torch.float32),
            fire_action.to(torch.float32),
            log_prob,
        ),
        dim=-1,
    )
    transfer_np = transfer.reshape(num_envs, num_agents, 4).cpu().numpy()

    result = {
        "course": transfer_np[..., 0].astype(np.int64),
        "target": transfer_np[..., 1].astype(np.int64),
        "fire": transfer_np[..., 2].astype(np.int64),
        "log_prob": transfer_np[..., 3],
        "next_h": out["next_h"].reshape(num_envs, num_agents, -1),
        "next_c": out["next_c"].reshape(num_envs, num_agents, -1),
    }
    if return_diagnostics:
        evaluated = evaluate_policy_heads(
            course_logits=out["course_logits"], target_logits=out["target_logits"],
            fire_logits=out["fire_logits"], course_action=course_action,
            target_action=target_action, fire_action=fire_action,
            entity_mask=entity_mask, alive_mask=alive,
            candidate_can_long=candidate_can_long, candidate_can_short=candidate_can_short,
            target_mask=flat_target_mask,
        )
        result.update({
            "log_prob": evaluated["log_prob"].reshape(num_envs, num_agents).cpu().numpy(),
            "entropy": evaluated["entropy"].reshape(num_envs, num_agents).cpu().numpy(),
            "head_log_prob": evaluated["head_log_prob"].reshape(num_envs, num_agents, 3).cpu().numpy(),
            "head_entropy": evaluated["head_entropy"].reshape(num_envs, num_agents, 3).cpu().numpy(),
            "fire_logits_selected": selected_fire_logits.reshape(num_envs, num_agents, 3).cpu().numpy(),
            "fire_mask": fire_mask.reshape(num_envs, num_agents, 3).cpu().numpy(),
            "target_mask": pointer_mask.reshape(num_envs, num_agents, -1).cpu().numpy(),
        })
    return result


@dataclass
class RolloutBatch:
    """One rollout with explicit per-agent signals and pre-action RNN states."""

    observations: dict[str, np.ndarray]
    recurrent_h: np.ndarray
    recurrent_c: np.ndarray
    course_action: np.ndarray
    target_action: np.ndarray
    fire_action: np.ndarray
    log_prob: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    value: np.ndarray
    next_value: np.ndarray
    episode_stats: list[dict[str, float]]

    def validate(self) -> None:
        expected = self.reward.shape
        if self.reward.ndim != 3:
            raise ValueError(f"reward must be [T,E,N], got {self.reward.shape}")
        for name in ("value", "log_prob", "course_action", "target_action", "fire_action"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {getattr(self, name).shape}")
        if self.recurrent_h.shape[:3] != expected or self.recurrent_c.shape[:3] != expected:
            raise ValueError("recurrent states must have leading [T,E,N] axes")
        if self.done.shape not in (expected[:2], expected):
            raise ValueError("done must be [T,E] or [T,E,N]")
        if self.next_value.shape != expected[1:]:
            raise ValueError(f"next_value must be [E,N], got {self.next_value.shape}")


def _broadcast_done(done: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    done_array = np.asarray(done)
    if done_array.shape == target_shape[:2]:
        done_array = np.broadcast_to(done_array[..., None], target_shape)
    elif done_array.shape != target_shape:
        raise ValueError(f"done must have shape {target_shape[:2]} or {target_shape}, got {done_array.shape}")
    return done_array.astype(np.float32, copy=False)


def compute_gae(
    reward: np.ndarray,
    value: np.ndarray,
    next_value: np.ndarray,
    done: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-agent GAE for tensors shaped ``[T, E, N]``."""
    reward = np.asarray(reward, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    next_value = np.asarray(next_value, dtype=np.float32)
    if reward.ndim != 3 or value.shape != reward.shape:
        raise ValueError("reward and value must have the same [T,E,N] shape")
    if next_value.shape != reward.shape[1:]:
        raise ValueError(f"next_value must have shape {reward.shape[1:]}, got {next_value.shape}")
    done_f = _broadcast_done(done, reward.shape)

    advantages = np.zeros_like(reward, dtype=np.float32)
    last_gae = np.zeros_like(next_value, dtype=np.float32)
    for step in reversed(range(reward.shape[0])):
        next_nonterminal = 1.0 - done_f[step]
        following_value = next_value if step == reward.shape[0] - 1 else value[step + 1]
        delta = reward[step] + gamma * following_value * next_nonterminal - value[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    return advantages, advantages + value
