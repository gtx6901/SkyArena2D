"""Action adapter for SkyArena2D training.

Converts MAPPO actor outputs to SkyArena-native actions.
"""
from __future__ import annotations

import numpy as np

from skyarena2d.adapters.action_types import SkyArenaSideAction
from skyarena2d.core.state import TeamState


class SkyArenaActionAdapter:
    """Converts discrete actor outputs to SkyArena-native actions.

    Actor outputs:
    - course_action: (N,) int — index into course_bins (relative offset from current heading)
    - search_goal_action: (N,) int — index into search_goal_grid (region ID)
    - target_action: (N,) int — 0=no target, 1..slots=slot index
    - fire_action: (N,) int — 0=no fire, 1=long, 2=short

    Decoding logic:
    - If has_active_contact: use course_action offset from current heading
    - Else: use search_goal_action to fly toward region center
    - Fire only if target_action > 0 and candidate_can_long/short allows
    """

    def __init__(
        self,
        candidate_slots: int = 6,
        course_bins: int = 16,
        search_goal_grid_size: int = 8,
        map_width: float = 3000.0,
        map_height: float = 4000.0,
        radar_freq: int = 1,
        jammer_freq: int = 1,
        use_jammer_strategy: bool = True,
        jammer_initial_silent: bool = True,
        jammer_on_after_contact: bool = True,
        jammer_range: float = 320.0,
        jammer_memory_steps: int = 10,
        max_jammers_per_side: int = 3,
    ) -> None:
        self.candidate_slots = candidate_slots
        self.course_bins = course_bins
        self.search_goal_grid_size = search_goal_grid_size
        self.map_width = map_width
        self.map_height = map_height
        self.radar_freq = radar_freq
        self.jammer_freq = jammer_freq
        self.use_jammer_strategy = use_jammer_strategy
        self.jammer_initial_silent = jammer_initial_silent
        self.jammer_on_after_contact = jammer_on_after_contact
        self.jammer_range = jammer_range
        self.jammer_memory_steps = jammer_memory_steps
        self.max_jammers_per_side = max_jammers_per_side
        self._ew_step_by_key: dict[object, int] = {}
        self._last_jammer_contact_by_key: dict[object, np.ndarray] = {}

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
        if entity_features is not None:
            entity_features = np.asarray(entity_features, dtype=np.float32)

        # Initialize output arrays
        course = np.zeros(N, dtype=np.float32)
        radar_freq_out = np.zeros(N, dtype=np.int32)
        jammer_freq_out = np.zeros(N, dtype=np.int32)
        fire_type = np.zeros(N, dtype=np.int32)
        target_idx = np.full(N, -1, dtype=np.int32)

        jammer_requests: list[tuple[int, float]] = []
        ew_step = self._resolve_ew_step(ew_state_key, step_count)
        last_jammer_contact = self._last_jammer_contact(ew_state_key, N)

        for i in range(N):
            if not own.alive[i]:
                continue

            # --- Course ---
            if has_active_contact[i]:
                # Use course_action offset from current heading
                offset_idx = int(course_action[i]) % self.course_bins
                offset = self._course_offsets[offset_idx]
                course[i] = (current_heading[i] + offset) % 360.0
            else:
                # Fly toward search_goal region center
                region_idx = int(search_goal_action[i]) % len(self._region_centers)
                cx, cy = self._region_centers[region_idx]
                ox, oy = float(own.pos[i, 0]), float(own.pos[i, 1])
                dx = cx - ox
                dy = cy - oy
                heading = np.degrees(np.arctan2(dy, dx)) % 360.0
                course[i] = heading

            # --- Radar fixed, jammer rule-based ---
            radar_freq_out[i] = self.radar_freq
            if not self.use_jammer_strategy:
                jammer_freq_out[i] = self.jammer_freq

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

            if self.use_jammer_strategy:
                request_distance = self._jammer_request_distance(
                    agent_idx=i,
                    candidate_ids=candidate_ids,
                    has_active_contact=has_active_contact,
                    entity_features=entity_features,
                )
                if request_distance is not None:
                    last_jammer_contact[i] = ew_step
                    jammer_requests.append((i, request_distance))
                elif self._should_hold_jammer(last_jammer_contact[i], ew_step):
                    jammer_requests.append((i, float(self.jammer_range)))

        if self.use_jammer_strategy:
            jammer_freq_out[:] = 0
            for i, _distance in self._select_jammer_requests(jammer_requests):
                jammer_freq_out[i] = int(max(self.jammer_freq, 1))

        return SkyArenaSideAction(
            course=course,
            radar_freq=radar_freq_out,
            jammer_freq=jammer_freq_out,
            fire_type=fire_type,
            target_idx=target_idx,
        )

    def reset_ew_state(self, ew_state_key: object | None = None) -> None:
        if ew_state_key is None:
            self._ew_step_by_key.clear()
            self._last_jammer_contact_by_key.clear()
            return
        self._ew_step_by_key.pop(ew_state_key, None)
        self._last_jammer_contact_by_key.pop(ew_state_key, None)

    def _resolve_ew_step(self, ew_state_key: object, step_count: int | None) -> int:
        if step_count is not None:
            step = int(step_count)
            self._ew_step_by_key[ew_state_key] = step
            return step
        step = int(self._ew_step_by_key.get(ew_state_key, 0)) + 1
        self._ew_step_by_key[ew_state_key] = step
        return step

    def _last_jammer_contact(self, ew_state_key: object, n: int) -> np.ndarray:
        arr = self._last_jammer_contact_by_key.get(ew_state_key)
        if arr is None or arr.shape != (n,):
            arr = np.full(n, -10**9, dtype=np.int32)
            self._last_jammer_contact_by_key[ew_state_key] = arr
        return arr

    def _jammer_request_distance(
        self,
        *,
        agent_idx: int,
        candidate_ids: np.ndarray,
        has_active_contact: np.ndarray,
        entity_features: np.ndarray | None,
    ) -> float | None:
        if not self.jammer_on_after_contact or not bool(has_active_contact[agent_idx]):
            return None
        valid = np.asarray(candidate_ids[agent_idx] >= 0, dtype=bool)
        if not np.any(valid):
            return None
        if entity_features is None:
            return 0.0
        distances = np.asarray(entity_features[agent_idx, :, 2], dtype=np.float32)
        valid_dist = distances[valid] * float(np.hypot(self.map_width, self.map_height))
        if valid_dist.size == 0:
            return None
        nearest = float(np.min(valid_dist))
        if nearest <= float(max(self.jammer_range, 0.0)):
            return nearest
        return None

    def _should_hold_jammer(self, last_contact_step: int, ew_step: int) -> bool:
        if not self.jammer_on_after_contact:
            return False
        return (int(ew_step) - int(last_contact_step)) <= max(int(self.jammer_memory_steps), 0)

    def _select_jammer_requests(self, requests: list[tuple[int, float]]) -> list[tuple[int, float]]:
        if not requests:
            return []
        max_jammers = max(int(self.max_jammers_per_side), 0)
        if max_jammers == 0:
            return []
        unique_by_agent: dict[int, tuple[int, float]] = {}
        for agent_idx, distance in requests:
            previous = unique_by_agent.get(agent_idx)
            if previous is None or distance < previous[1]:
                unique_by_agent[agent_idx] = (int(agent_idx), float(distance))
        return sorted(unique_by_agent.values(), key=lambda row: row[1])[:max_jammers]

    def build_masks(
        self,
        own: TeamState,
        fireable_long: np.ndarray,
        fireable_short: np.ndarray,
        candidate_ids: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build action masks for actor.

        Returns:
            course_mask: (N, course_bins) bool
            search_goal_mask: (N, grid_size*grid_size) bool
            target_mask: (N, candidate_slots+1) bool
        """
        N = own.num_fighters
        S = self.candidate_slots
        G = self.search_goal_grid_size

        course_mask = np.zeros((N, self.course_bins), dtype=bool)
        search_goal_mask = np.ones((N, G * G), dtype=bool)
        target_mask = np.zeros((N, S + 1), dtype=bool)

        for i in range(N):
            if own.alive[i]:
                course_mask[i, :] = True
            else:
                course_mask[i, 0] = True

            # Target mask: always can choose "no target" (index 0)
            target_mask[i, 0] = True
            for s in range(S):
                cid = int(candidate_ids[i, s])
                if cid >= 0:
                    target_mask[i, s + 1] = True

        return {
            "course_mask": course_mask,
            "search_goal_mask": search_goal_mask,
            "target_mask": target_mask,
        }

    def build_search_goal_lut(self) -> np.ndarray:
        """Build (grid_size*grid_size, 2) array of region center world coordinates."""
        return self._region_centers.copy()


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
