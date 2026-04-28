"""Rollout collection and GAE computation for SkyArena MAPPO.

Ported from MaCA-master/algo/rollout.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from .search_goal_manager import SearchGoalManager

FREE_COURSE = 0
SEARCH_GOAL = 1
NEAREST_VISIBLE = 2
SELECTED_TARGET = 3
INTERCEPT_TARGET = 4
ORBIT_LEFT = 5
ORBIT_RIGHT = 6
SUPPORT = 7
SEPARATION = 8

RESIDUAL_COURSE_INDICES = (24, 28, 30, 0, 2, 4, 8)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    invalid_logit = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, invalid_logit)
    return Categorical(logits=masked_logits)


def to_torch_batch(obs_batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "self_features": torch.as_tensor(obs_batch["self_features"], dtype=torch.float32, device=device),
        "entity_features": torch.as_tensor(obs_batch["entity_features"], dtype=torch.float32, device=device),
        "entity_mask": torch.as_tensor(obs_batch["entity_mask"], dtype=torch.bool, device=device),
        "semantic_map": torch.as_tensor(obs_batch["semantic_map"], dtype=torch.float32, device=device),
        "current_search_goal_id": torch.as_tensor(obs_batch["current_search_goal_id"], dtype=torch.long, device=device),
        "agent_id": torch.as_tensor(obs_batch["agent_id"], dtype=torch.long, device=device),
        "region_features": torch.as_tensor(obs_batch["region_features"], dtype=torch.float32, device=device),
        "course_mask": torch.as_tensor(obs_batch["course_mask"], dtype=torch.bool, device=device),
        "search_goal_mask": torch.as_tensor(obs_batch["search_goal_mask"], dtype=torch.bool, device=device),
        "target_mask": torch.as_tensor(obs_batch["target_mask"], dtype=torch.bool, device=device),
        "alive_mask": torch.as_tensor(obs_batch["alive_mask"], dtype=torch.float32, device=device),
        "has_active_contact": torch.as_tensor(obs_batch["has_active_contact"], dtype=torch.float32, device=device),
        "candidate_ids": torch.as_tensor(obs_batch["candidate_ids"], dtype=torch.long, device=device),
        "candidate_can_long": torch.as_tensor(obs_batch["candidate_can_long"], dtype=torch.bool, device=device),
        "candidate_can_short": torch.as_tensor(obs_batch["candidate_can_short"], dtype=torch.bool, device=device),
        "global_state": torch.as_tensor(obs_batch["global_state"], dtype=torch.float32, device=device),
    }


def stack_env_obs(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = obs_list[0].keys()
    return {key: np.stack([obs[key] for obs in obs_list], axis=0) for key in keys}


def allocate_batched_obs(example_obs: Dict[str, np.ndarray], num_envs: int) -> Dict[str, np.ndarray]:
    return {
        key: np.empty((num_envs, *value.shape), dtype=value.dtype)
        for key, value in example_obs.items()
    }


def allocate_rollout_obs(example_obs: Dict[str, np.ndarray], rollout_steps: int, num_envs: int) -> Dict[str, np.ndarray]:
    return {
        key: np.empty((rollout_steps, num_envs, *value.shape), dtype=value.dtype)
        for key, value in example_obs.items()
    }


def fill_batched_obs(dst_batch: Dict[str, np.ndarray], obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    for env_idx, obs in enumerate(obs_list):
        for key, dst in dst_batch.items():
            dst[env_idx] = obs[key]
    return dst_batch


def build_fire_mask_from_selected_targets(
    *,
    target_action: torch.Tensor,
    alive_mask: torch.Tensor,
    candidate_can_long: torch.Tensor,
    candidate_can_short: torch.Tensor,
) -> torch.Tensor:
    target_action = target_action.long()
    alive_mask = alive_mask.bool()
    valid_target = alive_mask & (target_action > 0)
    safe_slot_idx = torch.clamp(target_action - 1, min=0)
    selected_long = torch.gather(candidate_can_long.bool(), dim=1, index=safe_slot_idx.unsqueeze(-1)).squeeze(-1)
    selected_short = torch.gather(candidate_can_short.bool(), dim=1, index=safe_slot_idx.unsqueeze(-1)).squeeze(-1)
    fire_mask = torch.zeros((target_action.shape[0], 3), dtype=torch.bool, device=target_action.device)
    fire_mask[:, 0] = True
    fire_mask[:, 1] = valid_target & selected_long
    fire_mask[:, 2] = valid_target & selected_short
    return fire_mask


def build_movement_mode_mask(
    *,
    target_action: torch.Tensor,
    alive_mask: torch.Tensor,
    entity_mask: torch.Tensor,
    has_active_contact: torch.Tensor,
) -> torch.Tensor:
    target_action = target_action.long()
    alive = alive_mask.bool()
    has_entity = torch.any(entity_mask.bool(), dim=1)
    del has_active_contact
    mask = torch.zeros((target_action.shape[0], 9), dtype=torch.bool, device=target_action.device)
    mask[:, FREE_COURSE] = alive
    mask[:, SEARCH_GOAL] = alive
    mask[:, NEAREST_VISIBLE] = alive & has_entity
    selected = alive & (target_action > 0)
    mask[:, SELECTED_TARGET] = selected
    mask[:, INTERCEPT_TARGET] = selected
    mask[:, ORBIT_LEFT] = selected
    mask[:, ORBIT_RIGHT] = selected
    # SUPPORT needs per-env team contact and is filled by build_batched_movement_mode_mask.
    mask[:, SEPARATION] = alive
    return mask


def build_batched_movement_mode_mask(
    *,
    target_action: torch.Tensor,
    alive_mask: torch.Tensor,
    entity_mask: torch.Tensor,
    has_active_contact: torch.Tensor,
) -> torch.Tensor:
    leading_shape = target_action.shape
    flat = build_movement_mode_mask(
        target_action=target_action.reshape(-1),
        alive_mask=alive_mask.reshape(-1),
        entity_mask=entity_mask.reshape(-1, entity_mask.shape[-1]),
        has_active_contact=has_active_contact.reshape(-1),
    ).reshape(*leading_shape, 9)
    team_contact = torch.any(has_active_contact.bool() & alive_mask.bool(), dim=-1, keepdim=True)
    flat[..., SUPPORT] = alive_mask.bool() & team_contact
    return flat


def build_course_mask_for_movement_mode(
    *,
    movement_mode: torch.Tensor,
    alive_mask: torch.Tensor,
    course_bins: int,
) -> torch.Tensor:
    mode = movement_mode.long()
    alive = alive_mask.bool()
    mask = torch.zeros((*mode.shape, course_bins), dtype=torch.bool, device=mode.device)
    free = alive & (mode == FREE_COURSE)
    mask[free, :] = True
    non_free = alive & (mode != FREE_COURSE)
    residual = [idx for idx in RESIDUAL_COURSE_INDICES if idx < course_bins]
    if residual:
        mask[non_free.unsqueeze(-1).expand_as(mask)] = False
        mask[..., residual] |= non_free.unsqueeze(-1)
    mask[..., 0] |= ~alive
    return mask


def sample_policy_actions(
    actor,
    obs_batch: Dict[str, np.ndarray],
    hidden_state: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    deterministic: bool,
    *,
    search_goal_manager: Optional[SearchGoalManager] = None,
    return_diagnostics: bool = False,
):
    """Sample actions from actor given numpy obs batch."""
    obs_t = to_torch_batch(obs_batch, device)
    num_envs, num_agents = obs_batch["self_features"].shape[:2]
    flat_batch = {
        "self_features": obs_t["self_features"].reshape(num_envs * num_agents, -1),
        "entity_features": obs_t["entity_features"].reshape(num_envs * num_agents, *obs_t["entity_features"].shape[2:]),
        "entity_mask": obs_t["entity_mask"].reshape(num_envs * num_agents, -1),
        "semantic_map": obs_t["semantic_map"].reshape(num_envs * num_agents, *obs_t["semantic_map"].shape[2:]),
        "current_search_goal_id": obs_t["current_search_goal_id"].reshape(num_envs * num_agents),
        "agent_id": obs_t["agent_id"].reshape(num_envs * num_agents),
        "region_features": obs_t["region_features"].reshape(num_envs * num_agents, *obs_t["region_features"].shape[2:]),
    }
    h, c = hidden_state
    out = actor.step(flat_batch, (h.reshape(num_envs * num_agents, -1), c.reshape(num_envs * num_agents, -1)))

    base_course_mask = obs_t["course_mask"].reshape(num_envs * num_agents, -1)
    target_mask = obs_t["target_mask"].reshape(num_envs * num_agents, -1)
    alive_mask_t = obs_t["alive_mask"].reshape(num_envs * num_agents) > 0.5
    has_active_contact = obs_t["has_active_contact"].reshape(num_envs * num_agents) > 0.5

    target_dist = masked_categorical(out["target_logits"], target_mask)
    if deterministic:
        target_action = torch.argmax(out["target_logits"].masked_fill(~target_mask, torch.finfo(out["target_logits"].dtype).min), dim=-1)
    else:
        target_action = target_dist.sample()

    entity_mask_t = obs_t["entity_mask"].reshape(num_envs * num_agents, -1)
    has_entity_t = torch.any(entity_mask_t, dim=1)
    target_action = torch.where(alive_mask_t & has_entity_t, target_action, torch.zeros_like(target_action))

    movement_mode_mask = build_batched_movement_mode_mask(
        target_action=target_action.reshape(num_envs, num_agents),
        alive_mask=alive_mask_t.reshape(num_envs, num_agents),
        entity_mask=entity_mask_t.reshape(num_envs, num_agents, -1),
        has_active_contact=has_active_contact.reshape(num_envs, num_agents),
    ).reshape(num_envs * num_agents, -1)
    movement_mode_dist = masked_categorical(out["movement_mode_logits"], movement_mode_mask)
    if deterministic:
        movement_mode_action = torch.argmax(
            out["movement_mode_logits"].masked_fill(~movement_mode_mask, torch.finfo(out["movement_mode_logits"].dtype).min),
            dim=-1,
        )
    else:
        movement_mode_action = movement_mode_dist.sample()

    course_mask = build_course_mask_for_movement_mode(
        movement_mode=movement_mode_action,
        alive_mask=alive_mask_t,
        course_bins=out["course_logits"].shape[-1],
    ) & base_course_mask
    course_dist = masked_categorical(out["course_logits"], course_mask)
    if deterministic:
        course_action = torch.argmax(
            out["course_logits"].masked_fill(~course_mask, torch.finfo(out["course_logits"].dtype).min),
            dim=-1,
        )
    else:
        course_action = course_dist.sample()

    raw_search_goal_np = np.zeros((num_envs, num_agents), dtype=np.int64)
    executed_search_goal_np = raw_search_goal_np
    current_search_goal_id_np = np.zeros((num_envs, num_agents), dtype=np.int64)
    executed_search_goal_world_np = np.zeros((num_envs, num_agents, 2), dtype=np.float32)
    search_goal_refresh_mask_np = np.zeros((num_envs, num_agents), dtype=np.bool_)
    search_goal_planner_bias_np = np.zeros((num_envs, num_agents, out["search_goal_logits"].shape[-1]), dtype=np.float32)
    search_goal_log_prob = torch.zeros((num_envs * num_agents,), dtype=out["search_goal_logits"].dtype, device=device)

    if search_goal_manager is not None:
        refresh_mask_np = search_goal_manager.compute_refresh_mask(obs_batch)
        goal_mask_np = search_goal_manager.build_goal_mask(obs_batch)
        planner_bias_np = search_goal_manager.build_planner_bias(obs_batch)
        search_goal_planner_bias_np = planner_bias_np.astype(np.float32, copy=False)
        search_goal_mask = torch.as_tensor(goal_mask_np, dtype=torch.bool, device=device).reshape(num_envs * num_agents, -1)
        biased_search_goal_logits = out["search_goal_logits"] + torch.as_tensor(
            planner_bias_np.reshape(num_envs * num_agents, -1),
            dtype=out["search_goal_logits"].dtype,
            device=device,
        )
        search_goal_dist = masked_categorical(biased_search_goal_logits, search_goal_mask)
        if deterministic:
            raw_search_goal_action = torch.argmax(
                biased_search_goal_logits.masked_fill(~search_goal_mask, torch.finfo(biased_search_goal_logits.dtype).min),
                dim=-1,
            )
        else:
            raw_search_goal_action = search_goal_dist.sample()
        raw_search_goal_np = raw_search_goal_action.reshape(num_envs, num_agents).detach().cpu().numpy()
        goal_payload = search_goal_manager.apply(
            raw_goal_action=raw_search_goal_np,
            refresh_mask=refresh_mask_np,
            obs_batch=obs_batch,
        )
        executed_search_goal_np = goal_payload["executed_search_goal_action"]
        current_search_goal_id_np = goal_payload["current_search_goal_id"]
        executed_search_goal_world_np = goal_payload["executed_search_goal_world"]
        search_goal_refresh_mask_np = goal_payload["search_goal_refresh_mask"]
        refresh_t = torch.as_tensor(search_goal_refresh_mask_np.reshape(num_envs * num_agents), dtype=torch.bool, device=device)
        search_goal_log_prob = torch.where(
            refresh_t,
            search_goal_dist.log_prob(raw_search_goal_action),
            torch.zeros_like(search_goal_log_prob),
        )
    else:
        search_goal_mask = obs_t["search_goal_mask"].reshape(num_envs * num_agents, -1)
        search_goal_dist = masked_categorical(out["search_goal_logits"], search_goal_mask)
        if deterministic:
            raw_search_goal_action = torch.argmax(
                out["search_goal_logits"].masked_fill(~search_goal_mask, torch.finfo(out["search_goal_logits"].dtype).min),
                dim=-1,
            )
        else:
            raw_search_goal_action = search_goal_dist.sample()
        raw_search_goal_np = raw_search_goal_action.reshape(num_envs, num_agents).detach().cpu().numpy()
        executed_search_goal_np = raw_search_goal_np
        current_search_goal_id_np = executed_search_goal_np + 1
        search_goal_log_prob = search_goal_dist.log_prob(raw_search_goal_action)

    executed_search_goal_action = torch.as_tensor(
        executed_search_goal_np.reshape(num_envs * num_agents), dtype=torch.long, device=device,
    )

    candidate_can_long_t = obs_t["candidate_can_long"].reshape(num_envs * num_agents, -1)
    candidate_can_short_t = obs_t["candidate_can_short"].reshape(num_envs * num_agents, -1)
    has_target_opportunity_t = alive_mask_t & has_entity_t

    fire_mask_t = build_fire_mask_from_selected_targets(
        target_action=target_action,
        alive_mask=alive_mask_t,
        candidate_can_long=candidate_can_long_t,
        candidate_can_short=candidate_can_short_t,
    )
    fire_logits = out["fire_logits"][torch.arange(target_action.shape[0], device=device), target_action]
    fire_dist = masked_categorical(fire_logits, fire_mask_t)
    if deterministic:
        fire_action = torch.argmax(fire_logits.masked_fill(~fire_mask_t, torch.finfo(fire_logits.dtype).min), dim=-1)
    else:
        fire_action = fire_dist.sample()
    has_fire_opportunity_t = torch.any(fire_mask_t[:, 1:], dim=1)
    fire_action = torch.where(has_fire_opportunity_t, fire_action, torch.zeros_like(fire_action))

    movement_mode_log_prob = movement_mode_dist.log_prob(movement_mode_action)
    course_log_prob = course_dist.log_prob(course_action)
    movement_base_log_prob = movement_mode_log_prob + course_log_prob + search_goal_log_prob
    movement_log_prob = torch.where(alive_mask_t, movement_base_log_prob, torch.zeros_like(movement_base_log_prob))
    attack_log_prob = target_dist.log_prob(target_action) + fire_dist.log_prob(fire_action)
    total_log_prob = movement_log_prob + torch.where(
        has_target_opportunity_t, attack_log_prob, torch.zeros_like(attack_log_prob),
    )

    result = {
        "movement_mode": movement_mode_action.reshape(num_envs, num_agents).cpu().numpy(),
        "course": course_action.reshape(num_envs, num_agents).cpu().numpy(),
        "search_goal": executed_search_goal_np.astype(np.int64, copy=False),
        "current_search_goal_id": current_search_goal_id_np.astype(np.int64, copy=False),
        "search_goal_refresh_mask": search_goal_refresh_mask_np,
        "search_goal_planner_bias": search_goal_planner_bias_np,
        "search_goal_world": executed_search_goal_world_np,
        "target": target_action.reshape(num_envs, num_agents).cpu().numpy(),
        "fire": fire_action.reshape(num_envs, num_agents).cpu().numpy(),
        "log_prob": total_log_prob.reshape(num_envs, num_agents).detach().cpu().numpy(),
        "next_h": out["next_h"].reshape(num_envs, num_agents, -1),
        "next_c": out["next_c"].reshape(num_envs, num_agents, -1),
    }
    if return_diagnostics:
        result["fire_logits_selected"] = fire_logits.reshape(num_envs, num_agents, 3).detach().cpu().numpy()
        result["fire_mask"] = fire_mask_t.reshape(num_envs, num_agents, 3).cpu().numpy()
    return result


@dataclass
class RolloutBatch:
    observations: Dict[str, np.ndarray]
    initial_h: np.ndarray
    initial_c: np.ndarray
    movement_mode_action: np.ndarray
    course_action: np.ndarray
    decoded_course: np.ndarray
    search_goal_action: np.ndarray
    search_goal_refresh_mask: np.ndarray
    search_goal_planner_bias: np.ndarray
    target_action: np.ndarray
    fire_action: np.ndarray
    log_prob: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    value: np.ndarray
    next_value: np.ndarray
    episode_stats: List[Dict[str, float]]


def compute_gae(reward: np.ndarray, value: np.ndarray, next_value: np.ndarray, done: np.ndarray, gamma: float, gae_lambda: float):
    rollout_steps, num_envs = reward.shape
    advantages = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    last_gae = np.zeros((num_envs,), dtype=np.float32)
    for step in reversed(range(rollout_steps)):
        next_nonterminal = 1.0 - done[step]
        next_val = next_value if step == rollout_steps - 1 else value[step + 1]
        delta = reward[step] + gamma * next_val * next_nonterminal - value[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + value
    return advantages, returns
