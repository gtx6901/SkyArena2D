"""Training-side electronic-warfare heuristic.

This module is experimental. It is not an actor-learned action head.
It exists to keep radar/jammer decisions rule-based for now, reducing the
RL action-space complexity while MAPPO training stabilizes. Later it can be
replaced by a learned EW head or by a richer rule-based opponent strategy.
"""
from __future__ import annotations

import numpy as np


class EWHeuristicStrategy:
    """Rule-based electronic-warfare strategy used by training action decoding."""

    def __init__(
        self,
        radar_freq: int = 1,
        radar_freq_count: int = 10,
        radar_cycle_interval: int = 8,
        radar_stride: int = 3,
        jammer_freq: int = 1,
        jammer_cycle_interval: int = 6,
        jammer_stride: int = 7,
        jammer_barrage_prob: float = 0.0,
        enabled: bool = True,
        initial_silent: bool = True,
        on_after_contact: bool = True,
        jammer_range: float = 320.0,
        memory_steps: int = 10,
        max_jammers_per_side: int = 3,
        map_width: float = 3000.0,
        map_height: float = 4000.0,
    ) -> None:
        self.radar_freq = int(radar_freq)
        self.radar_freq_count = max(int(radar_freq_count), 1)
        self.radar_cycle_interval = max(int(radar_cycle_interval), 1)
        self.radar_stride = int(radar_stride)
        self.jammer_freq = int(jammer_freq)
        self.jammer_cycle_interval = max(int(jammer_cycle_interval), 1)
        self.jammer_stride = int(jammer_stride)
        self.jammer_barrage_prob = float(np.clip(jammer_barrage_prob, 0.0, 1.0))
        self.enabled = bool(enabled)
        self.initial_silent = bool(initial_silent)
        self.on_after_contact = bool(on_after_contact)
        self.jammer_range = float(jammer_range)
        self.memory_steps = int(memory_steps)
        self.max_jammers_per_side = int(max_jammers_per_side)
        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self._step_by_key: dict[object, int] = {}
        self._last_contact_by_key: dict[object, np.ndarray] = {}

    def reset(self, key: object | None = None) -> None:
        if key is None:
            self._step_by_key.clear()
            self._last_contact_by_key.clear()
            return
        self._step_by_key.pop(key, None)
        self._last_contact_by_key.pop(key, None)

    def compute(
        self,
        *,
        key: object,
        n_agents: int,
        alive: np.ndarray,
        candidate_ids: np.ndarray,
        has_active_contact: np.ndarray,
        entity_features: np.ndarray | None,
        step_count: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute rule-based radar and jammer frequencies.

        Returns:
            radar_freq: (n_agents,) int32
            jammer_freq: (n_agents,) int32
        """
        radar_freq = np.zeros((n_agents,), dtype=np.int32)
        jammer_freq = np.zeros((n_agents,), dtype=np.int32)

        alive = np.asarray(alive, dtype=bool)
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        has_active_contact = np.asarray(has_active_contact, dtype=bool)
        if entity_features is not None:
            entity_features = np.asarray(entity_features, dtype=np.float32)

        step = self._resolve_step(key, step_count)
        for agent_idx in range(n_agents):
            if agent_idx >= len(alive) or not bool(alive[agent_idx]):
                continue
            radar_freq[agent_idx] = self._radar_frequency(agent_idx, step)

        if not self.enabled:
            return radar_freq, jammer_freq

        last_contact = self._last_contact(key, n_agents)
        requests: list[tuple[int, float]] = []

        for agent_idx in range(n_agents):
            if agent_idx >= len(alive) or not bool(alive[agent_idx]):
                continue
            request_distance = self._jammer_request_distance(
                agent_idx=agent_idx,
                candidate_ids=candidate_ids,
                has_active_contact=has_active_contact,
                entity_features=entity_features,
            )
            if request_distance is not None:
                last_contact[agent_idx] = step
                requests.append((agent_idx, request_distance))
            elif self._should_hold_jammer(last_contact[agent_idx], step):
                requests.append((agent_idx, float(self.jammer_range)))

        for agent_idx, _distance in self._select_jammer_requests(requests):
            jammer_freq[agent_idx] = self._jammer_frequency(agent_idx, step)
        return radar_freq, jammer_freq

    def compute_jammer_freq(
        self,
        *,
        key: object,
        n_agents: int,
        alive: np.ndarray,
        candidate_ids: np.ndarray,
        has_active_contact: np.ndarray,
        entity_features: np.ndarray | None,
        step_count: int | None = None,
    ) -> np.ndarray:
        return self.compute(
            key=key,
            n_agents=n_agents,
            alive=alive,
            candidate_ids=candidate_ids,
            has_active_contact=has_active_contact,
            entity_features=entity_features,
            step_count=step_count,
        )[1]

    def compute_radar_freq(
        self,
        *,
        key: object,
        n_agents: int,
        alive: np.ndarray,
        candidate_ids: np.ndarray,
        has_active_contact: np.ndarray,
        entity_features: np.ndarray | None,
        step_count: int | None = None,
    ) -> np.ndarray:
        return self.compute(
            key=key,
            n_agents=n_agents,
            alive=alive,
            candidate_ids=candidate_ids,
            has_active_contact=has_active_contact,
            entity_features=entity_features,
            step_count=step_count,
        )[0]

    def _radar_frequency(self, agent_idx: int, step: int) -> int:
        phase = int(step) // self.radar_cycle_interval
        return 1 + ((int(agent_idx) * self.radar_stride + phase) % self.radar_freq_count)

    def _jammer_frequency(self, agent_idx: int, step: int) -> int:
        if self.jammer_barrage_prob > 0.0 and self._use_barrage(agent_idx, step):
            return self.radar_freq_count + 1
        phase = int(step) // self.jammer_cycle_interval
        return 1 + ((int(agent_idx) * self.jammer_stride + phase) % self.radar_freq_count)

    def _use_barrage(self, agent_idx: int, step: int) -> bool:
        bucket = ((int(agent_idx) + 1) * 1103515245 + int(step) * 12345) % 10000
        return (bucket / 10000.0) < self.jammer_barrage_prob

    def _resolve_step(self, key: object, step_count: int | None) -> int:
        if step_count is not None:
            step = int(step_count)
            self._step_by_key[key] = step
            return step
        step = int(self._step_by_key.get(key, 0)) + 1
        self._step_by_key[key] = step
        return step

    def _last_contact(self, key: object, n_agents: int) -> np.ndarray:
        arr = self._last_contact_by_key.get(key)
        if arr is None or arr.shape != (n_agents,):
            arr = np.full((n_agents,), -10**9, dtype=np.int32)
            self._last_contact_by_key[key] = arr
        return arr

    def _jammer_request_distance(
        self,
        *,
        agent_idx: int,
        candidate_ids: np.ndarray,
        has_active_contact: np.ndarray,
        entity_features: np.ndarray | None,
    ) -> float | None:
        if not self.on_after_contact or not bool(has_active_contact[agent_idx]):
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

    def _should_hold_jammer(self, last_contact_step: int, step: int) -> bool:
        if not self.on_after_contact:
            return False
        return (int(step) - int(last_contact_step)) <= max(int(self.memory_steps), 0)

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
