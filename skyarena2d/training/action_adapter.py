"""Decode the compact policy action into SkyArena's native side action."""
from __future__ import annotations

import numpy as np

from skyarena2d.adapters.action_types import SkyArenaSideAction
from skyarena2d.core.state import TeamState
from skyarena2d.training.ew_strategy import EWHeuristicStrategy

TURN_DELTAS_DEG = np.asarray(
    [-90.0, -45.0, -22.5, -10.0, 0.0, 10.0, 22.5, 45.0, 90.0],
    dtype=np.float32,
)


class SkyArenaActionAdapter:
    """Convert factorised policy actions to :class:`SkyArenaSideAction`.

    The learned contract contains only an atomic relative turn, an entity
    pointer target, and a weapon choice conditioned on that target. Radar and
    jamming stay in an explicit rule outside the learned action space.
    """

    def __init__(
        self,
        candidate_slots: int = 19,
        course_bins: int = 9,
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
        if int(course_bins) != len(TURN_DELTAS_DEG):
            raise ValueError(f"course_bins must be {len(TURN_DELTAS_DEG)} for atomic turns")
        self.candidate_slots = int(candidate_slots)
        self.course_bins = int(course_bins)
        self.radar_freq = int(radar_freq)
        self.jammer_freq = int(jammer_freq)
        self.ew_strategy = ew_strategy
        if use_jammer_strategy and self.ew_strategy is None:
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

    def decode(
        self,
        *,
        course_action: np.ndarray,
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
        n_agents = own.num_fighters
        course_action = np.asarray(course_action, dtype=np.int64)
        target_action = np.asarray(target_action, dtype=np.int64)
        fire_action = np.asarray(fire_action, dtype=np.int64)
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        candidate_can_long = np.asarray(candidate_can_long, dtype=bool)
        candidate_can_short = np.asarray(candidate_can_short, dtype=bool)
        has_active_contact = np.asarray(has_active_contact, dtype=bool)
        current_heading = np.asarray(current_heading, dtype=np.float32)

        vectors = {
            "course_action": course_action,
            "target_action": target_action,
            "fire_action": fire_action,
            "has_active_contact": has_active_contact,
            "current_heading": current_heading,
        }
        for name, value in vectors.items():
            if value.shape != (n_agents,):
                raise ValueError(f"{name} must have shape ({n_agents},), got {value.shape}")
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] != n_agents:
            raise ValueError("candidate_ids must have one row per fighter")
        if candidate_can_long.shape != candidate_ids.shape or candidate_can_short.shape != candidate_ids.shape:
            raise ValueError("candidate fire masks must match candidate_ids")

        alive = own.alive[:n_agents]
        turn_idx = np.mod(course_action, self.course_bins)
        course = current_heading.copy()
        course[alive] = (current_heading[alive] + TURN_DELTAS_DEG[turn_idx[alive]]) % 360.0

        fire_type = np.zeros(n_agents, dtype=np.int32)
        target_idx = np.full(n_agents, -1, dtype=np.int32)
        token_slots = candidate_ids.shape[1]
        for agent_idx in np.flatnonzero(alive):
            slot = int(target_action[agent_idx]) - 1
            if slot < 0 or slot >= token_slots:
                continue
            candidate_id = int(candidate_ids[agent_idx, slot])
            if candidate_id < 0:
                continue
            target_idx[agent_idx] = candidate_id
            requested_fire = int(fire_action[agent_idx])
            if requested_fire == 1 and candidate_can_long[agent_idx, slot]:
                fire_type[agent_idx] = 1
            elif requested_fire == 2 and candidate_can_short[agent_idx, slot]:
                fire_type[agent_idx] = 2

        radar_freq_out = np.zeros(n_agents, dtype=np.int32)
        jammer_freq_out = np.zeros(n_agents, dtype=np.int32)
        if self.ew_strategy is not None:
            radar_freq_out, jammer_freq_out = self.ew_strategy.compute(
                key=ew_state_key,
                n_agents=n_agents,
                alive=alive,
                candidate_ids=candidate_ids,
                has_active_contact=has_active_contact,
                entity_features=entity_features,
                step_count=step_count,
            )
        else:
            radar_freq_out[alive] = self.radar_freq
            jammer_freq_out[alive] = self.jammer_freq

        return SkyArenaSideAction(
            course=course.astype(np.float32, copy=False),
            radar_freq=radar_freq_out,
            jammer_freq=jammer_freq_out,
            fire_type=fire_type,
            target_idx=target_idx,
        )

    def reset_ew_state(self, ew_state_key: object | None = None) -> None:
        if self.ew_strategy is not None:
            self.ew_strategy.reset(ew_state_key)
