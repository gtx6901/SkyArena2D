"""Compact, agent-centric observations for SkyArena MAPPO.

The decentralized policy sees exact friendly state plus enemies that are either
currently visible or present in its short track memory. Every relative entity
feature is expressed in the observing fighter's heading-aligned frame. The
centralized critic keeps its separate full-truth state.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from skyarena2d.core.state import TeamState

MAX_FIGHTERS = 10
GLOBAL_STATE_DIM = 1 + MAX_FIGHTERS * 9 + MAX_FIGHTERS * 9
SELF_FEATURE_DIM = 22
ENTITY_FEATURE_DIM = 20


class SkyArenaTrainingObsBuilder:
    """Build compact fixed-shape policy observations without enemy leakage.

    Entity slots have a stable layout for a configured team size: all other
    friendly fighters first, then enemy slots indexed by enemy id. Friendly and
    padding slots use ``candidate_ids == -1`` and can never be selected by
    ``target_action``. Enemy slots remain selectable while their track is
    recent, even when they are not currently fireable.
    """

    def __init__(
        self,
        num_fighters: int,
        candidate_slots: int = 10,
        map_width: float = 3000.0,
        map_height: float = 4000.0,
        track_memory_steps: int = 10,
    ) -> None:
        self.num_fighters = int(num_fighters)
        self.enemy_slots = max(int(candidate_slots), self.num_fighters)
        self.ally_slots = max(self.num_fighters - 1, 0)
        self.entity_slots = self.ally_slots + self.enemy_slots
        self.candidate_slots = self.entity_slots
        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self.max_distance = float(np.hypot(self.map_width, self.map_height))
        self.track_memory_steps = int(track_memory_steps)
        self._track_memory: dict[int, dict[int, dict[str, Any]]] = {}

    def build_policy_obs(
        self,
        own: TeamState,
        enemy: TeamState,
        visible_matrix: np.ndarray,
        fireable_long: np.ndarray,
        fireable_short: np.ndarray,
        step_count: int,
        max_steps: int,
    ) -> dict[str, np.ndarray]:
        """Return decentralized policy inputs for every controlled fighter."""
        self._update_track_memory(enemy, visible_matrix, step_count)
        self_features = self._build_self_features(
            own, step_count / max(int(max_steps), 1)
        )
        (
            entity_features,
            entity_mask,
            candidate_ids,
            candidate_can_long,
            candidate_can_short,
        ) = self._build_entity_features(
            own,
            enemy,
            visible_matrix,
            fireable_long,
            fireable_short,
            step_count,
        )

        alive_mask = own.alive[: self.num_fighters].astype(np.float32, copy=True)
        visible = np.asarray(visible_matrix, dtype=bool)
        has_active_contact = np.zeros(self.num_fighters, dtype=np.float32)
        n_visible_rows = min(self.num_fighters, visible.shape[0])
        if n_visible_rows:
            has_active_contact[:n_visible_rows] = np.any(
                visible[:n_visible_rows], axis=1
            ).astype(np.float32)

        course_mask = np.zeros((self.num_fighters, 9), dtype=bool)
        course_mask[:, 0] = True
        course_mask[alive_mask.astype(bool), :] = True

        target_mask = np.zeros(
            (self.num_fighters, self.entity_slots + 1), dtype=bool
        )
        target_mask[:, 0] = True
        target_mask[:, 1:] = entity_mask & (candidate_ids >= 0)

        return {
            "self_features": self_features,
            "entity_features": entity_features,
            "entity_mask": entity_mask,
            "candidate_ids": candidate_ids,
            "candidate_can_long": candidate_can_long,
            "candidate_can_short": candidate_can_short,
            "agent_id": np.arange(self.num_fighters, dtype=np.int64),
            "alive_mask": alive_mask,
            "has_active_contact": has_active_contact,
            "course_mask": course_mask,
            "target_mask": target_mask,
        }

    def build_global_state(
        self,
        red: TeamState,
        blue: TeamState,
        step_count: int,
        max_steps: int,
    ) -> np.ndarray:
        """Build the 181-D centralized full-truth critic input."""
        state = np.zeros(GLOBAL_STATE_DIM, dtype=np.float32)
        state[0] = step_count / max(int(max_steps), 1)
        idx = _pack_team_features(
            state, 1, red, MAX_FIGHTERS, self.map_width, self.map_height
        )
        _pack_team_features(
            state, idx, blue, MAX_FIGHTERS, self.map_width, self.map_height
        )
        return state

    def reset(self) -> None:
        self._track_memory.clear()

    def _update_track_memory(
        self,
        enemy: TeamState,
        visible_matrix: np.ndarray,
        step_count: int,
    ) -> None:
        visible = np.asarray(visible_matrix, dtype=bool)
        own_count = min(self.num_fighters, visible.shape[0])
        enemy_count = min(enemy.pos.shape[0], visible.shape[1], self.enemy_slots)
        for observer_idx in range(own_count):
            memory = self._track_memory.setdefault(observer_idx, {})
            for enemy_idx in range(enemy_count):
                if not visible[observer_idx, enemy_idx]:
                    continue
                memory[enemy_idx] = {
                    "last_pos": enemy.pos[enemy_idx].copy(),
                    "last_step": int(step_count),
                    "alive": bool(enemy.alive[enemy_idx]),
                    "heading": float(enemy.heading[enemy_idx]),
                    "speed": float(enemy.speed[enemy_idx]),
                    "unit_type": int(enemy.unit_type[enemy_idx]),
                }

    def _build_self_features(
        self, own: TeamState, step_progress: float
    ) -> np.ndarray:
        feat = np.zeros((self.num_fighters, SELF_FEATURE_DIM), dtype=np.float32)
        n = min(self.num_fighters, own.pos.shape[0])
        if n == 0:
            return feat

        extent = max(self.map_width, self.map_height)
        heading = np.deg2rad(own.heading[:n])
        feat[:n, 0] = own.speed[:n] / 10.0
        feat[:n, 1] = np.cos(heading)
        feat[:n, 2] = np.sin(heading)
        feat[:n, 3] = own.long_ammo[:n] / 10.0
        feat[:n, 4] = own.short_ammo[:n] / 10.0
        feat[:n, 5] = own.radar_on[:n].astype(np.float32)
        feat[:n, 6] = own.radar_freq[:n] / 10.0
        feat[:n, 7] = own.jammer_on[:n].astype(np.float32)
        feat[:n, 8] = own.jammer_freq[:n] / 10.0
        feat[:n, 9] = own.alive[:n].astype(np.float32)
        feat[:n, 10] = own.unit_type[:n].astype(np.float32)
        feat[:n, 11] = float(step_progress)
        feat[:n, 12] = own.radar_range[:n] / extent
        feat[:n, 13] = own.long_range[:n] / extent
        feat[:n, 14] = own.short_range[:n] / extent
        feat[:n, 15] = own.long_hit_prob[:n]
        feat[:n, 16] = own.short_hit_prob[:n]
        feat[:n, 17] = np.clip(
            own.pos[:n, 0] / self.map_width, 0.0, 1.0
        )
        feat[:n, 18] = np.clip(
            (self.map_width - own.pos[:n, 0]) / self.map_width, 0.0, 1.0
        )
        feat[:n, 19] = np.clip(
            own.pos[:n, 1] / self.map_height, 0.0, 1.0
        )
        feat[:n, 20] = np.clip(
            (self.map_height - own.pos[:n, 1]) / self.map_height, 0.0, 1.0
        )
        feat[:n, 21] = np.arange(n, dtype=np.float32) / max(
            self.num_fighters - 1, 1
        )
        return feat

    def _build_entity_features(
        self,
        own: TeamState,
        enemy: TeamState,
        visible_matrix: np.ndarray,
        fireable_long: np.ndarray,
        fireable_short: np.ndarray,
        step_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        features = np.zeros(
            (self.num_fighters, self.entity_slots, ENTITY_FEATURE_DIM),
            dtype=np.float32,
        )
        entity_mask = np.zeros((self.num_fighters, self.entity_slots), dtype=bool)
        candidate_ids = np.full(
            (self.num_fighters, self.entity_slots), -1, dtype=np.int64
        )
        can_long = np.zeros((self.num_fighters, self.entity_slots), dtype=bool)
        can_short = np.zeros((self.num_fighters, self.entity_slots), dtype=bool)

        visible = np.asarray(visible_matrix, dtype=bool)
        fire_long = np.asarray(fireable_long, dtype=bool)
        fire_short = np.asarray(fireable_short, dtype=bool)
        own_count = min(self.num_fighters, own.pos.shape[0])
        enemy_count = min(enemy.pos.shape[0], self.enemy_slots)

        for observer_idx in range(own_count):
            if not own.alive[observer_idx]:
                continue
            slot = 0
            for ally_idx in range(own_count):
                if ally_idx == observer_idx:
                    continue
                if own.alive[ally_idx]:
                    self._write_entity_token(
                        features[observer_idx, slot],
                        own,
                        ally_idx,
                        own,
                        observer_idx,
                        is_ally=True,
                        is_visible=True,
                        is_tracked_only=False,
                        track_age=0.0,
                        token_can_long=False,
                        token_can_short=False,
                    )
                    entity_mask[observer_idx, slot] = True
                slot += 1

            memory = self._track_memory.get(observer_idx, {})
            for enemy_idx in range(enemy_count):
                enemy_slot = self.ally_slots + enemy_idx
                is_visible = (
                    observer_idx < visible.shape[0]
                    and enemy_idx < visible.shape[1]
                    and bool(visible[observer_idx, enemy_idx])
                )
                tracked = enemy_idx in memory and (
                    int(step_count) - int(memory[enemy_idx]["last_step"])
                    <= self.track_memory_steps
                )
                if not is_visible and not tracked:
                    continue

                if is_visible:
                    if not enemy.alive[enemy_idx]:
                        continue
                    source: TeamState | dict[str, Any] = enemy
                    track_age = 0.0
                else:
                    source = memory[enemy_idx]
                    if not source["alive"]:
                        continue
                    track_age = min(
                        (int(step_count) - int(source["last_step"]))
                        / max(self.track_memory_steps, 1),
                        1.0,
                    )

                token_can_long = bool(
                    is_visible
                    and observer_idx < fire_long.shape[0]
                    and enemy_idx < fire_long.shape[1]
                    and fire_long[observer_idx, enemy_idx]
                )
                token_can_short = bool(
                    is_visible
                    and observer_idx < fire_short.shape[0]
                    and enemy_idx < fire_short.shape[1]
                    and fire_short[observer_idx, enemy_idx]
                )
                self._write_entity_token(
                    features[observer_idx, enemy_slot],
                    source,
                    enemy_idx,
                    own,
                    observer_idx,
                    is_ally=False,
                    is_visible=is_visible,
                    is_tracked_only=not is_visible,
                    track_age=track_age,
                    token_can_long=token_can_long,
                    token_can_short=token_can_short,
                )
                entity_mask[observer_idx, enemy_slot] = True
                candidate_ids[observer_idx, enemy_slot] = enemy_idx
                can_long[observer_idx, enemy_slot] = token_can_long
                can_short[observer_idx, enemy_slot] = token_can_short

        return features, entity_mask, candidate_ids, can_long, can_short

    def _write_entity_token(
        self,
        out: np.ndarray,
        entity: TeamState | dict[str, Any],
        entity_idx: int,
        own: TeamState,
        observer_idx: int,
        *,
        is_ally: bool,
        is_visible: bool,
        is_tracked_only: bool,
        track_age: float,
        token_can_long: bool,
        token_can_short: bool,
    ) -> None:
        if isinstance(entity, TeamState):
            pos = entity.pos[entity_idx]
            heading = float(entity.heading[entity_idx])
            speed = float(entity.speed[entity_idx])
            alive = float(entity.alive[entity_idx])
            unit_type = float(entity.unit_type[entity_idx])
            long_ammo = (
                float(entity.long_ammo[entity_idx]) / 10.0 if is_ally else 0.0
            )
            short_ammo = (
                float(entity.short_ammo[entity_idx]) / 10.0 if is_ally else 0.0
            )
        else:
            pos = entity["last_pos"]
            heading = float(entity["heading"])
            speed = float(entity["speed"])
            alive = float(entity["alive"])
            unit_type = float(entity["unit_type"])
            long_ammo = 0.0
            short_ammo = 0.0

        delta = np.asarray(pos, dtype=np.float32) - own.pos[observer_idx]
        observer_heading = np.deg2rad(float(own.heading[observer_idx]))
        cos_h = float(np.cos(observer_heading))
        sin_h = float(np.sin(observer_heading))
        forward = cos_h * float(delta[0]) + sin_h * float(delta[1])
        left = -sin_h * float(delta[0]) + cos_h * float(delta[1])
        distance = float(np.hypot(delta[0], delta[1]))
        bearing_scale = max(distance, 1e-6)
        relative_heading = np.deg2rad(
            heading - float(own.heading[observer_idx])
        )

        out[0] = forward / self.max_distance
        out[1] = left / self.max_distance
        out[2] = distance / self.max_distance
        out[3] = forward / bearing_scale
        out[4] = left / bearing_scale
        out[5] = float(np.cos(relative_heading))
        out[6] = float(np.sin(relative_heading))
        out[7] = (speed - float(own.speed[observer_idx])) / 10.0
        out[8] = speed / 10.0
        out[9] = alive
        out[10] = unit_type
        out[11] = float(is_ally)
        out[12] = float(not is_ally)
        out[13] = float(is_visible and not is_ally)
        out[14] = float(is_tracked_only)
        out[15] = float(track_age)
        out[16] = long_ammo
        out[17] = short_ammo
        out[18] = float(token_can_long)
        out[19] = float(token_can_short)


def _pack_team_features(
    state: np.ndarray,
    idx: int,
    team: TeamState,
    max_fighters: int,
    map_width: float,
    map_height: float,
) -> int:
    n = min(team.num_fighters, max_fighters)
    for fighter_idx in range(max_fighters):
        if fighter_idx < n:
            state[idx] = team.pos[fighter_idx, 0] / map_width
            state[idx + 1] = team.pos[fighter_idx, 1] / map_height
            heading = np.deg2rad(float(team.heading[fighter_idx]))
            state[idx + 2] = float(np.cos(heading))
            state[idx + 3] = float(np.sin(heading))
            state[idx + 4] = float(team.alive[fighter_idx])
            state[idx + 5] = float(team.long_ammo[fighter_idx]) / 10.0
            state[idx + 6] = float(team.short_ammo[fighter_idx]) / 10.0
            state[idx + 7] = float(team.speed[fighter_idx]) / 10.0
            state[idx + 8] = float(team.unit_type[fighter_idx])
        idx += 9
    return idx
