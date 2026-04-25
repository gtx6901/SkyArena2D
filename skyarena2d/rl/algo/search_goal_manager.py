"""Search goal manager for SkyArena MAPPO.

Ported from MaCA-master/algo/search_goal_manager.py.
Manages per-agent persistent search goal assignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class TeamSearchPlannerConfig:
    num_envs: int
    num_agents: int
    map_size_x: float
    map_size_y: float
    search_goal_grid_size: int = 8
    goal_hold_steps: int = 8
    goal_reach_radius: float = 80.0
    recent_goal_cooldown: int = 3
    recent_goal_penalty: float = 8.0
    adjacent_goal_penalty: float = 3.0
    frontier_bonus_coef: float = 1.0
    support_quota: int = 2
    local_region_capacity: int = 2
    planner_contact_enter_steps: int = 3
    planner_contact_exit_steps: int = 6
    support_radius: float = 120.0
    opening_local_support_target: int = 2
    planner_support_bias_coef: float = 0.05
    planner_lone_bias_coef: float = 0.08


def _build_region_centers(grid_size: int, map_size_x: float, map_size_y: float) -> np.ndarray:
    """Build (G*G, 2) array of region center world coordinates."""
    cell_w = map_size_x / grid_size
    cell_h = map_size_y / grid_size
    centers = []
    for row in range(grid_size):
        for col in range(grid_size):
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            centers.append([cx, cy])
    return np.array(centers, dtype=np.float32)


class TeamSearchPlanner:
    """Batched team-level persistent assignment for global search regions."""

    def __init__(self, config: TeamSearchPlannerConfig):
        self.config = config
        self.num_envs = int(config.num_envs)
        self.num_agents = int(config.num_agents)
        self.map_size_x = float(max(config.map_size_x, 1.0))
        self.map_size_y = float(max(config.map_size_y, 1.0))
        self.grid_size = max(int(config.search_goal_grid_size), 1)
        self.goal_hold_steps = max(int(config.goal_hold_steps), 1)
        self.goal_reach_radius_sq = float(max(config.goal_reach_radius, 1.0)) ** 2
        self.recent_goal_cooldown = max(int(config.recent_goal_cooldown), 0)
        self.recent_goal_penalty = float(max(config.recent_goal_penalty, 0.0))
        self.adjacent_goal_penalty = float(max(config.adjacent_goal_penalty, 0.0))
        self.frontier_bonus_coef = float(config.frontier_bonus_coef)
        self.support_quota = max(int(config.support_quota), 0)
        self.local_region_capacity = max(int(config.local_region_capacity), 1)
        self.planner_contact_enter_steps = max(int(config.planner_contact_enter_steps), 1)
        self.planner_contact_exit_steps = max(int(config.planner_contact_exit_steps), 1)
        self.support_radius_sq = float(max(config.support_radius, 1.0)) ** 2
        self.opening_local_support_target = max(int(config.opening_local_support_target), 1)
        self.planner_support_bias_coef = float(max(config.planner_support_bias_coef, 0.0))
        self.planner_lone_bias_coef = float(max(config.planner_lone_bias_coef, 0.0))
        self.search_goal_bins = self.grid_size * self.grid_size
        self._region_centers = _build_region_centers(self.grid_size, self.map_size_x, self.map_size_y)
        self._adjacent_region_mask = self._build_adjacent_region_mask(self.grid_size)

        self.current_goal_id = np.full((self.num_envs, self.num_agents), -1, dtype=np.int64)
        self.goal_age = np.zeros((self.num_envs, self.num_agents), dtype=np.int32)
        self.goal_world = np.zeros((self.num_envs, self.num_agents, 2), dtype=np.float32)
        self.recent_goal_ids = np.full(
            (self.num_envs, self.num_agents, max(self.recent_goal_cooldown, 1)),
            -1,
            dtype=np.int64,
        )
        self.contact_streak = np.zeros((self.num_envs,), dtype=np.int32)
        self.no_contact_streak = np.zeros((self.num_envs,), dtype=np.int32)
        self.support_mode = np.zeros((self.num_envs,), dtype=np.bool_)
        self.contact_region_id = np.full((self.num_envs,), -1, dtype=np.int64)

    def reset_all(self) -> None:
        self.current_goal_id.fill(-1)
        self.goal_age.fill(0)
        self.goal_world.fill(0.0)
        self.recent_goal_ids.fill(-1)
        self.contact_streak.fill(0)
        self.no_contact_streak.fill(0)
        self.support_mode.fill(False)
        self.contact_region_id.fill(-1)

    def reset_envs(self, env_indices) -> None:
        if len(env_indices) == 0:
            return
        env_idx = np.asarray(env_indices, dtype=np.int64)
        self.current_goal_id[env_idx] = -1
        self.goal_age[env_idx] = 0
        self.goal_world[env_idx] = 0.0
        self.recent_goal_ids[env_idx] = -1
        self.contact_streak[env_idx] = 0
        self.no_contact_streak[env_idx] = 0
        self.support_mode[env_idx] = False
        self.contact_region_id[env_idx] = -1

    def clear_contact_goals(self, obs_batch: Dict[str, np.ndarray]) -> None:
        alive = np.asarray(obs_batch["alive_mask"], dtype=np.float32) > 0.5
        has_contact = np.asarray(obs_batch["has_active_contact"], dtype=np.float32) > 0.5
        clear_mask = (~alive) | has_contact
        self.current_goal_id[clear_mask] = -1
        self.goal_age[clear_mask] = 0
        self.goal_world[clear_mask] = 0.0

    def current_goal_observation(self) -> np.ndarray:
        active = self.current_goal_id >= 0
        return np.where(active, self.current_goal_id + 1, 0).astype(np.int64)

    def apply(
        self,
        *,
        region_logits: np.ndarray,
        obs_batch: Dict[str, np.ndarray],
        raw_goal_action: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        del raw_goal_action
        logits = np.asarray(region_logits, dtype=np.float32)
        if logits.shape[:2] != (self.num_envs, self.num_agents):
            raise ValueError(
                f"region_logits shape must start with {(self.num_envs, self.num_agents)}, got {logits.shape}"
            )

        self.clear_contact_goals(obs_batch)

        self_features = np.asarray(obs_batch["self_features"], dtype=np.float32)
        alive = np.asarray(obs_batch["alive_mask"], dtype=np.float32) > 0.5
        has_contact = np.asarray(obs_batch["has_active_contact"], dtype=np.float32) > 0.5
        search_active = alive & (~has_contact)

        pos_x = self_features[..., 0] * self.map_size_x
        pos_y = self_features[..., 1] * self.map_size_y
        active_goal = self.current_goal_id >= 0
        goal_dx = self.goal_world[..., 0] - pos_x
        goal_dy = self.goal_world[..., 1] - pos_y
        goal_reached = active_goal & ((goal_dx * goal_dx + goal_dy * goal_dy) <= self.goal_reach_radius_sq)
        goal_expired = active_goal & (self.goal_age >= self.goal_hold_steps)
        refresh_mask = search_active & ((~active_goal) | goal_reached | goal_expired)

        raw_region_features = obs_batch.get("region_features")
        region_features = None if raw_region_features is None else np.asarray(raw_region_features, dtype=np.float32)
        self._update_contact_modes(has_contact, region_features)

        for env_idx in range(self.num_envs):
            agents = np.nonzero(refresh_mask[env_idx])[0]
            if agents.size == 0:
                continue
            held_mask = search_active[env_idx] & (~refresh_mask[env_idx]) & (self.current_goal_id[env_idx] >= 0)
            used_regions = set(int(r) for r in self.current_goal_id[env_idx, held_mask] if int(r) >= 0)
            neighborhood_occupancy = self._build_neighborhood_occupancy(used_regions)
            scores = logits[env_idx, agents].copy()
            self._apply_recent_cooldown(scores, env_idx, agents)
            order = np.argsort(-np.max(scores, axis=1))
            for local_idx in order:
                agent_idx = int(agents[local_idx])
                agent_scores = scores[local_idx].copy()
                assigned = self._best_region_with_capacity(
                    agent_scores, used_regions, neighborhood_occupancy,
                    local_capacity=self.local_region_capacity,
                )
                used_regions.add(assigned)
                neighborhood_occupancy += self._adjacent_region_mask[assigned].astype(np.int16, copy=False)
                self.current_goal_id[env_idx, agent_idx] = assigned
                self.goal_world[env_idx, agent_idx] = self._region_centers[assigned]
                self.goal_age[env_idx, agent_idx] = 0
                self._push_recent_goal(env_idx, agent_idx, assigned)

        hold_mask = search_active & (~refresh_mask) & (self.current_goal_id >= 0)
        self.goal_age[hold_mask] += 1
        self.current_goal_id[~search_active] = -1
        self.goal_age[~search_active] = 0
        self.goal_world[~search_active] = 0.0

        executed_goal_action = np.where(search_active & (self.current_goal_id >= 0), self.current_goal_id, 0).astype(np.int64)
        executed_goal_world = np.where(search_active[..., None], self.goal_world, 0.0).astype(np.float32)
        return {
            "executed_search_goal_action": executed_goal_action,
            "executed_search_goal_world": executed_goal_world,
            "current_search_goal_id": self.current_goal_observation(),
            "search_goal_refresh_mask": refresh_mask.astype(np.bool_, copy=False),
        }

    def _update_contact_modes(self, has_contact: np.ndarray, region_features: Optional[np.ndarray]) -> None:
        team_contact = np.any(has_contact, axis=1)
        self.contact_streak[team_contact] += 1
        self.contact_streak[~team_contact] = 0
        self.no_contact_streak[~team_contact] += 1
        self.no_contact_streak[team_contact] = 0
        enter_mask = self.contact_streak >= self.planner_contact_enter_steps
        exit_mask = self.no_contact_streak >= self.planner_contact_exit_steps
        self.support_mode[enter_mask] = True
        self.support_mode[exit_mask] = False

    def _apply_recent_cooldown(self, scores: np.ndarray, env_idx: int, agents: np.ndarray) -> None:
        if self.recent_goal_cooldown <= 0:
            return
        recent = self.recent_goal_ids[env_idx, agents]
        for local_idx in range(recent.shape[0]):
            valid_recent = recent[local_idx][recent[local_idx] >= 0]
            if valid_recent.size == 0:
                continue
            scores[local_idx, valid_recent] -= self.recent_goal_penalty

    @staticmethod
    def _best_region_with_capacity(
        score_row: np.ndarray,
        used_regions: set,
        neighborhood_occupancy: np.ndarray,
        *,
        local_capacity: int,
    ) -> int:
        order = np.argsort(-score_row)
        for region_id in order:
            rid = int(region_id)
            if rid not in used_regions and neighborhood_occupancy[rid] < local_capacity:
                return rid
        for region_id in order:
            rid = int(region_id)
            if rid not in used_regions:
                return rid
        return int(order[0])

    def _build_neighborhood_occupancy(self, used_regions: set) -> np.ndarray:
        occupancy = np.zeros((self.search_goal_bins,), dtype=np.int16)
        if not used_regions:
            return occupancy
        used = np.asarray(list(used_regions), dtype=np.int64)
        return np.sum(self._adjacent_region_mask[used], axis=0, dtype=np.int16)

    def _push_recent_goal(self, env_idx: int, agent_idx: int, region_id: int) -> None:
        if self.recent_goal_cooldown <= 0:
            return
        self.recent_goal_ids[env_idx, agent_idx, 1:] = self.recent_goal_ids[env_idx, agent_idx, :-1]
        self.recent_goal_ids[env_idx, agent_idx, 0] = int(region_id)

    @staticmethod
    def _build_adjacent_region_mask(grid_size: int) -> np.ndarray:
        bins = max(int(grid_size), 1) ** 2
        mask = np.zeros((bins, bins), dtype=np.bool_)
        for region_id in range(bins):
            row = region_id // grid_size
            col = region_id % grid_size
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = row + dr
                    cc = col + dc
                    if 0 <= rr < grid_size and 0 <= cc < grid_size:
                        mask[region_id, rr * grid_size + cc] = True
        return mask


# Aliases for compatibility
SearchGoalManagerConfig = TeamSearchPlannerConfig
SearchGoalManager = TeamSearchPlanner
