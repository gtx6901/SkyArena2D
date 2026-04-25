"""Training observation builder for SkyArena2D.

Produces MAPPO-compatible observations:
- build_policy_obs: per-agent observations (NO enemy leak for invisible enemies)
- build_global_state: full-truth state for centralized critic
"""
from __future__ import annotations

from typing import Any

import numpy as np

from skyarena2d.core.state import TeamState

# Max fighters per side for global state padding
_MAX_FIGHTERS = 10
_GLOBAL_STATE_DIM = 1 + _MAX_FIGHTERS * 9 + _MAX_FIGHTERS * 9  # 181


class SkyArenaTrainingObsBuilder:
    """Builds MAPPO-compatible observations from SkyArena2D state.

    Enemy leak prevention:
    - Policy obs only includes enemies visible via visible_matrix or in track memory.
    - Invisible enemies with no track record are completely excluded.
    - Critic global_state contains full truth (all positions).
    """

    def __init__(
        self,
        num_fighters: int,
        candidate_slots: int = 6,
        search_goal_grid_size: int = 8,
        map_width: float = 3000.0,
        map_height: float = 4000.0,
        semantic_map_size: int = 100,
        track_memory_steps: int = 10,
    ) -> None:
        self.num_fighters = num_fighters
        self.candidate_slots = candidate_slots
        self.search_goal_grid_size = search_goal_grid_size
        self.map_width = map_width
        self.map_height = map_height
        self.semantic_map_size = semantic_map_size
        self.track_memory_steps = track_memory_steps

        # Track memory: per-agent, per-enemy — agent_idx -> enemy_idx -> {last_pos, last_step, ...}
        # Each agent only tracks enemies it has personally observed.
        self._track_memory: dict[int, dict[int, dict[str, Any]]] = {}

        # Precompute course offset LUT (16 bins)
        self._course_offsets = _build_course_offset_lut(16)

        # Precompute region centers LUT
        self._region_centers = _build_region_centers(
            search_goal_grid_size, map_width, map_height
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_policy_obs(
        self,
        own: TeamState,
        enemy: TeamState,
        visible_matrix: np.ndarray,
        fireable_long: np.ndarray,
        fireable_short: np.ndarray,
        step_count: int,
        max_steps: int,
        current_search_goal_id: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build per-agent policy observations.

        Args:
            own: Friendly TeamState (N fighters).
            enemy: Enemy TeamState (M fighters).
            visible_matrix: (N, M) bool — True if own[i] can see enemy[j].
            fireable_long: (N, M) bool — can fire long missile at enemy[j].
            fireable_short: (N, M) bool — can fire short missile at enemy[j].
            step_count: Current step.
            max_steps: Episode max steps.
            current_search_goal_id: (N,) int64 — current search goal region per agent.

        Returns:
            Dict of observation arrays.
        """
        N = self.num_fighters
        S = self.candidate_slots
        G = self.search_goal_grid_size
        M = self.semantic_map_size

        # Update track memory for newly visible enemies
        self._update_track_memory(enemy, visible_matrix, step_count)

        step_progress = step_count / max(max_steps, 1)

        # --- self_features (N, 20) ---
        self_features = self._build_self_features(own, step_progress)

        # --- entity_features, entity_mask, candidate_ids, candidate_can_long/short ---
        (
            entity_features,
            entity_mask,
            candidate_ids,
            candidate_can_long,
            candidate_can_short,
        ) = self._build_entity_features(
            own, enemy, visible_matrix, fireable_long, fireable_short, step_count
        )

        # --- semantic_map (N, 9, M, M) ---
        semantic_map = self._build_semantic_map(own, enemy, visible_matrix)

        # --- region_features (N, G*G, 10) ---
        region_features = self._build_region_features(
            own, enemy, visible_matrix, step_count
        )

        # --- alive_mask (N,) ---
        alive_mask = own.alive[: N].astype(np.float32)

        # --- has_active_contact (N,) ---
        has_active_contact = np.any(visible_matrix[:N, :], axis=1).astype(np.float32)

        # --- course_mask (N, 16) ---
        course_mask = np.zeros((N, 16), dtype=bool)
        for i in range(N):
            if own.alive[i]:
                course_mask[i, :] = True
            else:
                course_mask[i, 0] = True

        # --- search_goal_mask (N, G*G) ---
        search_goal_mask = np.ones((N, G * G), dtype=bool)

        # --- target_mask (N, S+1) ---
        target_mask = np.zeros((N, S + 1), dtype=bool)
        target_mask[:, 0] = True  # always can choose "no target"
        for i in range(N):
            for s in range(S):
                if entity_mask[i, s]:
                    target_mask[i, s + 1] = True

        # --- agent_id (N,) ---
        agent_id = np.arange(N, dtype=np.int64)

        # --- current_search_goal_id (N,) ---
        csg = np.asarray(current_search_goal_id, dtype=np.int64)
        if csg.shape != (N,):
            csg = np.zeros(N, dtype=np.int64)

        return {
            "self_features": self_features,
            "entity_features": entity_features,
            "entity_mask": entity_mask,
            "candidate_ids": candidate_ids,
            "candidate_can_long": candidate_can_long,
            "candidate_can_short": candidate_can_short,
            "semantic_map": semantic_map,
            "current_search_goal_id": csg,
            "agent_id": agent_id,
            "region_features": region_features,
            "alive_mask": alive_mask,
            "has_active_contact": has_active_contact,
            "course_mask": course_mask,
            "search_goal_mask": search_goal_mask,
            "target_mask": target_mask,
        }

    def build_global_state(
        self,
        red: TeamState,
        blue: TeamState,
        step_count: int,
        max_steps: int,
    ) -> np.ndarray:
        """Build global state for centralized critic (full truth, no visibility filter).

        Returns:
            (181,) float32 array.
        """
        state = np.zeros(_GLOBAL_STATE_DIM, dtype=np.float32)
        idx = 0

        # step progress
        state[idx] = step_count / max(max_steps, 1)
        idx += 1

        # red fighters (up to _MAX_FIGHTERS)
        idx = _pack_team_features(state, idx, red, _MAX_FIGHTERS, self.map_width, self.map_height)

        # blue fighters (up to _MAX_FIGHTERS)
        idx = _pack_team_features(state, idx, blue, _MAX_FIGHTERS, self.map_width, self.map_height)

        return state

    def reset(self) -> None:
        """Reset per-agent track memory at episode start."""
        self._track_memory.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_track_memory(
        self,
        enemy: TeamState,
        visible_matrix: np.ndarray,
        step_count: int,
    ) -> None:
        """Update per-agent track memory when enemies become visible to each own agent."""
        N_own = visible_matrix.shape[0]
        M_enemy = enemy.pos.shape[0]
        for i in range(N_own):
            if i not in self._track_memory:
                self._track_memory[i] = {}
            for j in range(M_enemy):
                if visible_matrix[i, j]:
                    self._track_memory[i][j] = {
                        "last_pos": enemy.pos[j].copy(),
                        "last_step": step_count,
                        "alive": bool(enemy.alive[j]),
                        "speed": float(enemy.speed[j]),
                        "unit_type": int(enemy.unit_type[j]),
                    }

    def _build_self_features(
        self, own: TeamState, step_progress: float
    ) -> np.ndarray:
        """Build (N, 20) self feature array."""
        N = self.num_fighters
        feat = np.zeros((N, 20), dtype=np.float32)

        n = min(N, own.pos.shape[0])
        # Normalize positions to [0, 1]
        feat[:n, 0] = own.pos[:n, 0] / self.map_width
        feat[:n, 1] = own.pos[:n, 1] / self.map_height
        # Heading as cos/sin
        heading_rad = np.deg2rad(own.heading[:n])
        feat[:n, 2] = np.cos(heading_rad)
        feat[:n, 3] = np.sin(heading_rad)
        # Speed (normalized by a rough max of 10)
        feat[:n, 4] = own.speed[:n] / 10.0
        # Ammo (normalized by max 10)
        feat[:n, 5] = own.long_ammo[:n] / 10.0
        feat[:n, 6] = own.short_ammo[:n] / 10.0
        # Radar on/freq
        feat[:n, 7] = own.radar_on[:n].astype(np.float32)
        feat[:n, 8] = own.radar_freq[:n] / 10.0
        # Jammer on/freq
        feat[:n, 9] = own.jammer_on[:n].astype(np.float32)
        feat[:n, 10] = own.jammer_freq[:n] / 10.0
        # Alive
        feat[:n, 11] = own.alive[:n].astype(np.float32)
        # Unit type
        feat[:n, 12] = own.unit_type[:n].astype(np.float32)
        # Step progress (same for all)
        feat[:n, 13] = step_progress
        # Radar range (normalized)
        feat[:n, 14] = own.radar_range[:n] / max(self.map_width, self.map_height)
        # Long/short range (normalized)
        feat[:n, 15] = own.long_range[:n] / max(self.map_width, self.map_height)
        feat[:n, 16] = own.short_range[:n] / max(self.map_width, self.map_height)
        # Hit probs
        feat[:n, 17] = own.long_hit_prob[:n]
        feat[:n, 18] = own.short_hit_prob[:n]
        # kills/losses encoded as 0 (team-level, not per-agent)
        feat[:n, 19] = 0.0

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
        """Build entity features with strict enemy leak prevention.

        Only includes enemy j for agent i if:
        - visible_matrix[i, j] is True, OR
        - enemy j is in track memory and last seen within track_memory_steps

        Returns:
            entity_features: (N, S, 10)
            entity_mask: (N, S) bool
            candidate_ids: (N, S) int64
            candidate_can_long: (N, S) bool
            candidate_can_short: (N, S) bool
        """
        N = self.num_fighters
        S = self.candidate_slots
        M = enemy.pos.shape[0]

        entity_features = np.zeros((N, S, 10), dtype=np.float32)
        entity_mask = np.zeros((N, S), dtype=bool)
        candidate_ids = np.full((N, S), -1, dtype=np.int64)
        candidate_can_long = np.zeros((N, S), dtype=bool)
        candidate_can_short = np.zeros((N, S), dtype=bool)

        max_dist = np.sqrt(self.map_width ** 2 + self.map_height ** 2)

        for i in range(N):
            if not own.alive[i]:
                continue

            slot = 0
            for j in range(M):
                if slot >= S:
                    break

                visible = bool(visible_matrix[i, j]) if i < visible_matrix.shape[0] and j < visible_matrix.shape[1] else False
                agent_mem = self._track_memory.get(i, {})
                tracked = (
                    j in agent_mem
                    and (step_count - agent_mem[j]["last_step"]) <= self.track_memory_steps
                )

                # CRITICAL: skip if neither visible nor tracked by this specific agent
                if not visible and not tracked:
                    continue

                # Use actual position if visible, last known if only tracked
                if visible:
                    ex, ey = float(enemy.pos[j, 0]), float(enemy.pos[j, 1])
                    e_alive = float(enemy.alive[j])
                    e_speed = float(enemy.speed[j])
                    e_type = float(enemy.unit_type[j])
                    is_tracked_only = 0.0
                else:
                    mem = agent_mem[j]
                    ex, ey = float(mem["last_pos"][0]), float(mem["last_pos"][1])
                    e_alive = float(mem["alive"])
                    e_speed = float(mem["speed"])
                    e_type = float(mem["unit_type"])
                    is_tracked_only = 1.0

                ox, oy = float(own.pos[i, 0]), float(own.pos[i, 1])
                dx = ex - ox
                dy = ey - oy
                dist = np.sqrt(dx * dx + dy * dy)
                bearing = np.degrees(np.arctan2(dy, dx)) % 360.0

                can_long = bool(fireable_long[i, j]) if i < fireable_long.shape[0] and j < fireable_long.shape[1] else False
                can_short = bool(fireable_short[i, j]) if i < fireable_short.shape[0] and j < fireable_short.shape[1] else False

                # hit prob (use own's hit prob as proxy)
                hit_prob = float(own.long_hit_prob[i]) if can_long else float(own.short_hit_prob[i]) if can_short else 0.0

                entity_features[i, slot, 0] = dx / self.map_width
                entity_features[i, slot, 1] = dy / self.map_height
                entity_features[i, slot, 2] = dist / max_dist
                entity_features[i, slot, 3] = bearing / 360.0
                entity_features[i, slot, 4] = e_alive
                entity_features[i, slot, 5] = e_type
                entity_features[i, slot, 6] = e_speed / 10.0
                entity_features[i, slot, 7] = float(can_long)
                entity_features[i, slot, 8] = float(can_short)
                entity_features[i, slot, 9] = hit_prob

                entity_mask[i, slot] = True
                candidate_ids[i, slot] = j
                candidate_can_long[i, slot] = can_long
                candidate_can_short[i, slot] = can_short
                slot += 1

        return entity_features, entity_mask, candidate_ids, candidate_can_long, candidate_can_short

    def _build_semantic_map(
        self,
        own: TeamState,
        enemy: TeamState,
        visible_matrix: np.ndarray,
    ) -> np.ndarray:
        """Build (N, 9, M, M) semantic map.

        Channels:
            0: own_alive
            1: own_heading_cos
            2: own_heading_sin
            3: enemy_visible (only visible enemies)
            4: enemy_passive (placeholder, zeros)
            5: ally_alive
            6: ally_heading_cos
            7: ally_heading_sin
            8: fireable_density (placeholder, zeros)
        """
        N = self.num_fighters
        M = self.semantic_map_size
        maps = np.zeros((N, 9, M, M), dtype=np.float32)

        def world_to_grid(x: float, y: float) -> tuple[int, int]:
            gx = int(np.clip(x / self.map_width * M, 0, M - 1))
            gy = int(np.clip(y / self.map_height * M, 0, M - 1))
            return gx, gy

        n_own = min(N, own.pos.shape[0])
        n_enemy = enemy.pos.shape[0]

        for i in range(N):
            if not own.alive[i]:
                continue

            # Own agent position
            gx, gy = world_to_grid(own.pos[i, 0], own.pos[i, 1])
            maps[i, 0, gy, gx] = 1.0  # own_alive
            h_rad = np.deg2rad(own.heading[i])
            maps[i, 1, gy, gx] = float(np.cos(h_rad))
            maps[i, 2, gy, gx] = float(np.sin(h_rad))

            # Allies (other own agents)
            for k in range(n_own):
                if k == i or not own.alive[k]:
                    continue
                agx, agy = world_to_grid(own.pos[k, 0], own.pos[k, 1])
                maps[i, 5, agy, agx] = 1.0
                ah_rad = np.deg2rad(own.heading[k])
                maps[i, 6, agy, agx] = float(np.cos(ah_rad))
                maps[i, 7, agy, agx] = float(np.sin(ah_rad))

            # Enemies — ONLY visible ones (no leak)
            for j in range(n_enemy):
                vis = bool(visible_matrix[i, j]) if i < visible_matrix.shape[0] and j < visible_matrix.shape[1] else False
                if vis and enemy.alive[j]:
                    egx, egy = world_to_grid(enemy.pos[j, 0], enemy.pos[j, 1])
                    maps[i, 3, egy, egx] = 1.0  # enemy_visible channel only

        return maps

    def _build_region_features(
        self,
        own: TeamState,
        enemy: TeamState,
        visible_matrix: np.ndarray,
        step_count: int,
    ) -> np.ndarray:
        """Build (N, G*G, 10) region features.

        Per-region features:
            0: center_x (normalized)
            1: center_y (normalized)
            2: has_own
            3: has_visible_enemy
            4: has_passive_enemy (placeholder)
            5: visit_recency (placeholder, 0)
            6: fireable_density (placeholder, 0)
            7: ally_count (normalized)
            8: enemy_count (normalized, only visible)
            9: distance_to_region (normalized)
        """
        N = self.num_fighters
        G = self.search_goal_grid_size
        num_regions = G * G
        feat = np.zeros((N, num_regions, 10), dtype=np.float32)

        cell_w = self.map_width / G
        cell_h = self.map_height / G
        max_dist = np.sqrt(self.map_width ** 2 + self.map_height ** 2)

        n_own = min(N, own.pos.shape[0])
        n_enemy = enemy.pos.shape[0]

        # Precompute which region each unit belongs to
        def pos_to_region(x: float, y: float) -> int:
            col = int(np.clip(x / cell_w, 0, G - 1))
            row = int(np.clip(y / cell_h, 0, G - 1))
            return row * G + col

        # Region centers
        centers = self._region_centers  # (G*G, 2)

        # Fill static region center features
        feat[:, :, 0] = centers[:, 0] / self.map_width
        feat[:, :, 1] = centers[:, 1] / self.map_height

        for i in range(N):
            if not own.alive[i]:
                continue

            ox, oy = float(own.pos[i, 0]), float(own.pos[i, 1])

            # Distance from agent i to each region center
            dx = centers[:, 0] - ox
            dy = centers[:, 1] - oy
            dists = np.sqrt(dx * dx + dy * dy)
            feat[i, :, 9] = dists / max_dist

            # Own agent in region
            own_region = pos_to_region(ox, oy)
            feat[i, own_region, 2] = 1.0

            # Allies in regions
            ally_counts = np.zeros(num_regions, dtype=np.float32)
            for k in range(n_own):
                if k == i or not own.alive[k]:
                    continue
                r = pos_to_region(float(own.pos[k, 0]), float(own.pos[k, 1]))
                ally_counts[r] += 1.0
            feat[i, :, 7] = ally_counts / max(n_own, 1)

            # Visible enemies in regions (NO leak)
            enemy_counts = np.zeros(num_regions, dtype=np.float32)
            for j in range(n_enemy):
                vis = bool(visible_matrix[i, j]) if i < visible_matrix.shape[0] and j < visible_matrix.shape[1] else False
                if vis and enemy.alive[j]:
                    r = pos_to_region(float(enemy.pos[j, 0]), float(enemy.pos[j, 1]))
                    enemy_counts[r] += 1.0
                    feat[i, r, 3] = 1.0  # has_visible_enemy

            feat[i, :, 8] = enemy_counts / max(n_enemy, 1)

        return feat


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


def _pack_team_features(
    state: np.ndarray,
    idx: int,
    team: TeamState,
    max_fighters: int,
    map_width: float,
    map_height: float,
) -> int:
    """Pack team fighter features into global state array. Returns updated idx."""
    n = min(team.num_fighters, max_fighters)
    for k in range(max_fighters):
        if k < n:
            state[idx + 0] = team.pos[k, 0] / map_width
            state[idx + 1] = team.pos[k, 1] / map_height
            h_rad = np.deg2rad(float(team.heading[k]))
            state[idx + 2] = float(np.cos(h_rad))
            state[idx + 3] = float(np.sin(h_rad))
            state[idx + 4] = float(team.alive[k])
            state[idx + 5] = float(team.long_ammo[k]) / 10.0
            state[idx + 6] = float(team.short_ammo[k]) / 10.0
            state[idx + 7] = float(team.speed[k]) / 10.0
            state[idx + 8] = float(team.unit_type[k])
        # else: zeros (already initialized)
        idx += 9
    return idx
