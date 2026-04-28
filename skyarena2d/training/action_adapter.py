"""Action adapter for SkyArena2D training.

Converts MAPPO actor outputs to SkyArena-native actions.
"""
from __future__ import annotations

import numpy as np

from skyarena2d.adapters.action_types import SkyArenaSideAction
from skyarena2d.core.state import TeamState
from skyarena2d.training.ew_strategy import EWHeuristicStrategy


class SkyArenaActionAdapter:
    """Converts discrete actor outputs to SkyArena-native actions.

    Actor outputs:
    - movement_mode_action: (N,) int — Movement V3 mode
    - course_action: (N,) int — absolute heading bin or residual offset bin
    - search_goal_action: (N,) int — index into search_goal_grid (region ID)
    - target_action: (N,) int — 0=no target, 1..slots=slot index
    - fire_action: (N,) int — 0=no fire, 1=long, 2=short

    Decoding logic:
    - FREE_COURSE decodes course_action as an absolute heading bin
    - Other modes decode course_action as a residual offset from a reference bearing
    - Fire only if target_action > 0 and candidate_can_long/short allows
    """

    def __init__(
        self,
        candidate_slots: int = 6,
        course_bins: int = 32,
        search_goal_grid_size: int = 8,
        map_width: float = 3000.0,
        map_height: float = 4000.0,
        radar_freq: int = 1,
        radar_freq_count: int = 10,
        radar_cycle_interval: int = 8,
        radar_stride: int = 3,
        jammer_freq: int = 1,
        jammer_cycle_interval: int = 6,
        jammer_stride: int = 7,
        jammer_barrage_prob: float = 0.0,
        use_jammer_strategy: bool = True,
        jammer_initial_silent: bool = True,
        jammer_on_after_contact: bool = True,
        jammer_range: float = 320.0,
        jammer_memory_steps: int = 10,
        max_jammers_per_side: int = 3,
        ew_strategy: EWHeuristicStrategy | None = None,
    ) -> None:
        self.candidate_slots = candidate_slots
        self.course_bins = course_bins
        self.search_goal_grid_size = search_goal_grid_size
        self.map_width = map_width
        self.map_height = map_height
        self.radar_freq = radar_freq
        self.jammer_freq = jammer_freq
        self.use_jammer_strategy = use_jammer_strategy
        self.ew_strategy = ew_strategy
        if self.use_jammer_strategy and self.ew_strategy is None:
            self.ew_strategy = EWHeuristicStrategy(
                radar_freq=radar_freq,
                radar_freq_count=radar_freq_count,
                radar_cycle_interval=radar_cycle_interval,
                radar_stride=radar_stride,
                jammer_freq=jammer_freq,
                jammer_cycle_interval=jammer_cycle_interval,
                jammer_stride=jammer_stride,
                jammer_barrage_prob=jammer_barrage_prob,
                enabled=True,
                initial_silent=jammer_initial_silent,
                on_after_contact=jammer_on_after_contact,
                jammer_range=jammer_range,
                memory_steps=jammer_memory_steps,
                max_jammers_per_side=max_jammers_per_side,
                map_width=map_width,
                map_height=map_height,
            )

        # Precompute course offset LUT
        self._course_offsets = _build_course_offset_lut(course_bins)

        # Precompute search goal region centers
        self._region_centers = _build_region_centers(
            search_goal_grid_size, map_width, map_height
        )

    def decode(
        self,
        course_action: np.ndarray,
        search_goal_action: np.ndarray,
        target_action: np.ndarray,
        fire_action: np.ndarray,
        own: TeamState,
        candidate_ids: np.ndarray,
        candidate_can_long: np.ndarray,
        candidate_can_short: np.ndarray,
        has_active_contact: np.ndarray,
        current_heading: np.ndarray,
        movement_mode_action: np.ndarray | None = None,
        enemy: TeamState | None = None,
        entity_features: np.ndarray | None = None,
        ew_state_key: object = "default",
        step_count: int | None = None,
    ) -> SkyArenaSideAction:
        """Decode actor outputs to SkyArena-native action.

        Args:
            course_action: (N,) int — course bin index
            search_goal_action: (N,) int — search goal region index
            target_action: (N,) int — 0=no target, 1..slots=slot index
            fire_action: (N,) int — 0=no fire, 1=long, 2=short
            own: TeamState
            candidate_ids: (N, slots) int64 — enemy indices
            candidate_can_long: (N, slots) bool
            candidate_can_short: (N, slots) bool
            has_active_contact: (N,) bool — True if any visible enemy
            current_heading: (N,) float32 — current heading degrees
            movement_mode_action: optional (N,) int — Movement V3 mode

        Returns:
            SkyArenaSideAction with decoded actions.
        """
        N = own.num_fighters

        # Ensure inputs are numpy arrays
        course_action = np.asarray(course_action, dtype=np.int32)
        search_goal_action = np.asarray(search_goal_action, dtype=np.int32)
        target_action = np.asarray(target_action, dtype=np.int32)
        fire_action = np.asarray(fire_action, dtype=np.int32)
        has_active_contact = np.asarray(has_active_contact, dtype=bool)
        current_heading = np.asarray(current_heading, dtype=np.float32)
        if movement_mode_action is None:
            has_entity = np.any(np.asarray(candidate_ids) >= 0, axis=1)
            movement_mode_action = np.where(has_entity, 2, 1).astype(np.int32, copy=False)
        else:
            movement_mode_action = np.asarray(movement_mode_action, dtype=np.int32)

        # Initialize output arrays
        course = np.zeros(N, dtype=np.float32)
        radar_freq_out = np.zeros(N, dtype=np.int32)
        jammer_freq_out = np.zeros(N, dtype=np.int32)
        fire_type = np.zeros(N, dtype=np.int32)
        target_idx = np.full(N, -1, dtype=np.int32)

        for i in range(N):
            if not own.alive[i]:
                continue

            # --- Course ---
            mode = int(movement_mode_action[i]) % 9
            if mode == 0:
                course[i] = _absolute_heading_from_bin(int(course_action[i]), self.course_bins)
            else:
                reference = self._movement_reference_bearing(
                    agent_idx=i,
                    movement_mode_action=mode,
                    search_goal_action=int(search_goal_action[i]),
                    target_action=int(target_action[i]),
                    own=own,
                    enemy=enemy,
                    candidate_ids=candidate_ids,
                    candidate_can_long=candidate_can_long,
                    candidate_can_short=candidate_can_short,
                    has_active_contact=has_active_contact,
                    entity_features=entity_features,
                )
                offset_idx = int(course_action[i]) % self.course_bins
                offset = self._course_offsets[offset_idx]
                course[i] = (reference + offset) % 360.0

            # --- Target and Fire ---
            tgt_act = int(target_action[i])
            if tgt_act == 0:
                # No target
                fire_type[i] = 0
                target_idx[i] = -1
            else:
                # Target slot (1-indexed in action, 0-indexed in array)
                slot = tgt_act - 1
                if slot < 0 or slot >= self.candidate_slots:
                    fire_type[i] = 0
                    target_idx[i] = -1
                    continue

                cid = int(candidate_ids[i, slot])
                if cid < 0:
                    # Invalid candidate
                    fire_type[i] = 0
                    target_idx[i] = -1
                    continue

                target_idx[i] = cid

                # Fire type (check if allowed)
                fire_act = int(fire_action[i])
                if fire_act == 1 and candidate_can_long[i, slot]:
                    fire_type[i] = 1
                elif fire_act == 2 and candidate_can_short[i, slot]:
                    fire_type[i] = 2
                else:
                    fire_type[i] = 0

        if self.ew_strategy is not None:
            radar_freq_out, jammer_freq_out = self.ew_strategy.compute(
                key=ew_state_key,
                n_agents=N,
                alive=own.alive[:N],
                candidate_ids=candidate_ids,
                has_active_contact=has_active_contact,
                entity_features=entity_features,
                step_count=step_count,
            )
        else:
            radar_freq_out[own.alive[:N]] = int(self.radar_freq)
            jammer_freq_out[own.alive[:N]] = int(self.jammer_freq)

        return SkyArenaSideAction(
            course=course,
            radar_freq=radar_freq_out,
            jammer_freq=jammer_freq_out,
            fire_type=fire_type,
            target_idx=target_idx,
        )

    def reset_ew_state(self, ew_state_key: object | None = None) -> None:
        if self.ew_strategy is not None:
            self.ew_strategy.reset(ew_state_key)

    def _movement_reference_bearing(
        self,
        *,
        agent_idx: int,
        movement_mode_action: int,
        search_goal_action: int,
        target_action: int,
        own: TeamState,
        enemy: TeamState | None,
        candidate_ids: np.ndarray,
        candidate_can_long: np.ndarray,
        candidate_can_short: np.ndarray,
        has_active_contact: np.ndarray,
        entity_features: np.ndarray | None,
    ) -> float:
        i = int(agent_idx)
        mode = int(movement_mode_action) % 9
        if mode == 1:
            return self._search_goal_bearing(i, search_goal_action, own)
        if mode == 2:
            bearing = self._nearest_entity_bearing(i, candidate_ids, entity_features)
            return bearing if bearing is not None else self._search_goal_bearing(i, search_goal_action, own)
        if mode == 3:
            bearing = self._selected_target_bearing(i, target_action, candidate_ids, entity_features)
            if bearing is None:
                bearing = self._nearest_entity_bearing(i, candidate_ids, entity_features)
            return bearing if bearing is not None else self._search_goal_bearing(i, search_goal_action, own)
        if mode == 4:
            bearing = self._selected_intercept_bearing(i, target_action, own, enemy, candidate_ids, entity_features)
            if bearing is None:
                bearing = self._nearest_entity_bearing(i, candidate_ids, entity_features)
            return bearing if bearing is not None else self._search_goal_bearing(i, search_goal_action, own)
        if mode in (5, 6):
            bearing = self._selected_target_bearing(i, target_action, candidate_ids, entity_features)
            if bearing is None:
                bearing = self._nearest_entity_bearing(i, candidate_ids, entity_features)
            if bearing is None:
                return self._search_goal_bearing(i, search_goal_action, own)
            delta = -90.0 if mode == 5 else 90.0
            return float((bearing + delta) % 360.0)
        if mode == 7:
            bearing = self._ally_contact_support_bearing(i, own, has_active_contact)
            return bearing if bearing is not None else self._search_goal_bearing(i, search_goal_action, own)
        if mode == 8:
            bearing = self._separation_bearing(i, own, candidate_ids, entity_features)
            return bearing if bearing is not None else self._search_goal_bearing(i, search_goal_action, own)
        return self._search_goal_bearing(i, search_goal_action, own)

    def _search_goal_bearing(self, agent_idx: int, search_goal_action: int, own: TeamState) -> float:
        region_idx = int(search_goal_action) % len(self._region_centers)
        cx, cy = self._region_centers[region_idx]
        ox, oy = float(own.pos[agent_idx, 0]), float(own.pos[agent_idx, 1])
        return _bearing_from_delta(cx - ox, cy - oy)

    def _selected_target_bearing(
        self,
        agent_idx: int,
        target_action: int,
        candidate_ids: np.ndarray,
        entity_features: np.ndarray | None,
    ) -> float | None:
        slot = int(target_action) - 1
        if slot < 0 or slot >= self.candidate_slots:
            return None
        if int(candidate_ids[agent_idx, slot]) < 0 or entity_features is None:
            return None
        return float(entity_features[agent_idx, slot, 3] * 360.0) % 360.0

    def _nearest_entity_bearing(
        self,
        agent_idx: int,
        candidate_ids: np.ndarray,
        entity_features: np.ndarray | None,
    ) -> float | None:
        if entity_features is None:
            return None
        valid = candidate_ids[agent_idx] >= 0
        slots = np.nonzero(valid)[0]
        if slots.size == 0:
            return None
        dist = entity_features[agent_idx, slots, 2]
        slot = int(slots[int(np.argmin(dist))])
        return float(entity_features[agent_idx, slot, 3] * 360.0) % 360.0

    def _selected_intercept_bearing(
        self,
        agent_idx: int,
        target_action: int,
        own: TeamState,
        enemy: TeamState | None,
        candidate_ids: np.ndarray,
        entity_features: np.ndarray | None,
    ) -> float | None:
        slot = int(target_action) - 1
        if slot < 0 or slot >= self.candidate_slots:
            return None
        target_idx = int(candidate_ids[agent_idx, slot])
        if target_idx < 0:
            return None
        if enemy is None or target_idx >= enemy.pos.shape[0]:
            return self._selected_target_bearing(agent_idx, target_action, candidate_ids, entity_features)
        ox, oy = float(own.pos[agent_idx, 0]), float(own.pos[agent_idx, 1])
        tx, ty = float(enemy.pos[target_idx, 0]), float(enemy.pos[target_idx, 1])
        dx = tx - ox
        dy = ty - oy
        dist = float(np.hypot(dx, dy))
        own_speed = max(float(own.speed[agent_idx]), 1e-3)
        horizon = float(np.clip(dist / own_speed, 1.0, 20.0))
        theta = np.deg2rad(float(enemy.heading[target_idx]))
        pred_x = tx + np.cos(theta) * float(enemy.speed[target_idx]) * horizon
        pred_y = ty + np.sin(theta) * float(enemy.speed[target_idx]) * horizon
        return _bearing_from_delta(pred_x - ox, pred_y - oy)

    def _nearest_fireable_bearing(
        self,
        agent_idx: int,
        candidate_ids: np.ndarray,
        candidate_can_long: np.ndarray,
        candidate_can_short: np.ndarray,
        entity_features: np.ndarray | None,
    ) -> float | None:
        if entity_features is None:
            return None
        valid = candidate_ids[agent_idx] >= 0
        fireable = candidate_can_long[agent_idx] | candidate_can_short[agent_idx]
        slots = np.nonzero(valid & fireable)[0]
        if slots.size == 0:
            return None
        dist = entity_features[agent_idx, slots, 2]
        slot = int(slots[int(np.argmin(dist))])
        return float(entity_features[agent_idx, slot, 3] * 360.0) % 360.0

    @staticmethod
    def _ally_contact_support_bearing(agent_idx: int, own: TeamState, has_active_contact: np.ndarray) -> float | None:
        alive_contact = own.alive[: own.num_fighters] & has_active_contact[: own.num_fighters]
        if not np.any(alive_contact):
            return None
        positions = own.pos[: own.num_fighters][alive_contact]
        centroid = np.mean(positions, axis=0)
        ox, oy = float(own.pos[agent_idx, 0]), float(own.pos[agent_idx, 1])
        dx = float(centroid[0]) - ox
        dy = float(centroid[1]) - oy
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        return _bearing_from_delta(dx, dy)

    def _separation_bearing(
        self,
        agent_idx: int,
        own: TeamState,
        candidate_ids: np.ndarray,
        entity_features: np.ndarray | None,
    ) -> float | None:
        i = int(agent_idx)
        ox, oy = float(own.pos[i, 0]), float(own.pos[i, 1])
        away = np.zeros(2, dtype=np.float32)

        for j in range(own.num_fighters):
            if j == i or not own.alive[j]:
                continue
            dx = ox - float(own.pos[j, 0])
            dy = oy - float(own.pos[j, 1])
            dist_sq = max(dx * dx + dy * dy, 1.0)
            away += np.array([dx, dy], dtype=np.float32) / dist_sq

        if entity_features is not None:
            for slot in range(min(self.candidate_slots, candidate_ids.shape[1])):
                if int(candidate_ids[i, slot]) < 0:
                    continue
                bearing = float(entity_features[i, slot, 3] * 360.0)
                dist = max(float(entity_features[i, slot, 2]), 1e-3)
                theta = np.deg2rad(bearing)
                away -= np.array([np.cos(theta), np.sin(theta)], dtype=np.float32) / dist

        norm = float(np.linalg.norm(away))
        if norm < 1e-6:
            return None
        return _bearing_from_delta(float(away[0]), float(away[1]))

    def _map_center_bearing(self, agent_idx: int, own: TeamState) -> float:
        ox, oy = float(own.pos[agent_idx, 0]), float(own.pos[agent_idx, 1])
        return _bearing_from_delta(self.map_width * 0.5 - ox, self.map_height * 0.5 - oy)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _build_course_offset_lut(n_bins: int = 16) -> np.ndarray:
    """Build course offset LUT.

    bin 0 = 0 (straight ahead)
    bins 1..n//2-1 = right turns (+22.5, +45, ...)
    bin n//2 = 180 (U-turn)
    bins n//2+1..n-1 = left turns (-157.5, ..., -22.5)
    """
    offsets = np.zeros(n_bins, dtype=np.float32)
    step = 360.0 / n_bins
    for k in range(n_bins):
        if k == 0:
            offsets[k] = 0.0
        elif k <= n_bins // 2:
            offsets[k] = k * step
        else:
            offsets[k] = (k - n_bins) * step
    return offsets


def _build_region_centers(
    grid_size: int, map_width: float, map_height: float
) -> np.ndarray:
    """Build (G*G, 2) array of region center world coordinates."""
    cell_w = map_width / grid_size
    cell_h = map_height / grid_size
    centers = []
    for row in range(grid_size):
        for col in range(grid_size):
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            centers.append([cx, cy])
    return np.array(centers, dtype=np.float32)


def _bearing_from_delta(dx: float, dy: float) -> float:
    return float(np.degrees(np.arctan2(dy, dx)) % 360.0)


def _absolute_heading_from_bin(action: int, n_bins: int) -> float:
    return float((int(action) % max(int(n_bins), 1)) * (360.0 / max(int(n_bins), 1)))
