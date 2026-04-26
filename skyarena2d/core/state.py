from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

UNIT_FIGHTER = 0
UNIT_DETECTOR = 1


@dataclass(slots=True)
class MissileEvent:
    attacker_side: Literal["red", "blue"]
    attacker_idx: int
    target_side: Literal["red", "blue"]
    target_idx: int
    missile_type: Literal["long", "short"]
    launch_step: int
    resolve_step: int
    hit_prob: float


@dataclass(slots=True)
class LaunchRecord:
    attacker_side: Literal["red", "blue"]
    attacker_idx: int
    target_side: Literal["red", "blue"]
    target_idx: int
    missile_type: Literal["long", "short"]
    hit_prob: float
    valid: bool
    reason: str


@dataclass(slots=True)
class TeamState:
    side: Literal["red", "blue"]
    num_fighters: int
    num_detectors: int
    alive: np.ndarray
    pos: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    unit_type: np.ndarray
    radar_on: np.ndarray
    radar_freq: np.ndarray
    jammer_on: np.ndarray
    jammer_freq: np.ndarray
    long_ammo: np.ndarray
    short_ammo: np.ndarray
    radar_range: np.ndarray
    radar_fov_deg: np.ndarray
    jammer_range: np.ndarray
    long_range: np.ndarray
    short_range: np.ndarray
    long_hit_prob: np.ndarray
    short_hit_prob: np.ndarray
    last_action: np.ndarray
    last_reward: np.ndarray
    kills: int = 0
    losses: int = 0

    @property
    def total_units(self) -> int:
        return self.num_fighters + self.num_detectors

    @property
    def fighter_indices(self) -> np.ndarray:
        return np.arange(self.num_fighters, dtype=np.int64)

    @property
    def detector_indices(self) -> np.ndarray:
        return np.arange(self.num_fighters, self.total_units, dtype=np.int64)

    @property
    def fighter_alive_count(self) -> int:
        if self.num_fighters == 0:
            return 0
        return int(np.count_nonzero(self.alive[: self.num_fighters]))

    @property
    def detector_alive_count(self) -> int:
        if self.num_detectors == 0:
            return 0
        return int(np.count_nonzero(self.alive[self.num_fighters :]))

    @property
    def alive_count(self) -> int:
        return int(np.count_nonzero(self.alive))

    def remaining_missiles(self) -> int:
        if self.total_units == 0:
            return 0
        return int(np.sum(self.long_ammo + self.short_ammo))


@dataclass(slots=True)
class MetricsTracker:
    first_contact_step: int | None = None
    first_fire_opportunity_step: int | None = None
    missiles_launched_long: int = 0
    missiles_launched_short: int = 0
    missiles_hit: int = 0
    missiles_missed: int = 0
    jammed_detection_count: int = 0
    passive_detection_count: int = 0
    unique_targets_engaged_red: set[int] = field(default_factory=set)
    unique_targets_engaged_blue: set[int] = field(default_factory=set)
    overkill_values_red: list[int] = field(default_factory=list)
    overkill_values_blue: list[int] = field(default_factory=list)
    # --- new: fireable/attempted/selected tracking ---
    fire_attempts_red: int = 0
    fire_attempts_blue: int = 0
    fire_opportunities_red: int = 0   # step-agent count with fireable opportunity
    fire_opportunities_blue: int = 0
    fire_executions_red: int = 0      # step-agent count that fired given opportunity
    fire_executions_blue: int = 0
    invalid_fire_count_red: int = 0
    invalid_fire_count_blue: int = 0
    unique_selected_targets_red: set[int] = field(default_factory=set)
    unique_selected_targets_blue: set[int] = field(default_factory=set)
    selected_overkill_values_red: list[int] = field(default_factory=list)
    selected_overkill_values_blue: list[int] = field(default_factory=list)
    selected_expected_exchange_acc: float = 0.0
    selected_expected_exchange_steps: int = 0


@dataclass(slots=True)
class StepCache:
    red_visible: np.ndarray
    blue_visible: np.ndarray
    red_jammed: np.ndarray
    blue_jammed: np.ndarray
    red_fireable_long: np.ndarray
    red_fireable_short: np.ndarray
    blue_fireable_long: np.ndarray
    blue_fireable_short: np.ndarray
    red_passive: list[list[dict[str, Any]]]
    blue_passive: list[list[dict[str, Any]]]
    launch_records: list[LaunchRecord]
    resolved_records: list[dict[str, Any]]
    strike_list: list[dict[str, Any]]
    # --- new: attempted/selected per-agent arrays ---
    red_attempted_fire: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    blue_attempted_fire: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    red_selected_long: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    red_selected_short: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    blue_selected_long: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    blue_selected_short: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    red_selected_target_idx: np.ndarray = field(default_factory=lambda: np.full(0, -1, dtype=np.int32))
    blue_selected_target_idx: np.ndarray = field(default_factory=lambda: np.full(0, -1, dtype=np.int32))
    red_attempted_long_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    red_attempted_short_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    blue_attempted_long_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    blue_attempted_short_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    red_selected_long_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    red_selected_short_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    blue_selected_long_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    blue_selected_short_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    reward_components: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EnvState:
    red: TeamState
    blue: TeamState
    step_count: int
    episode_idx: int
    rng: np.random.Generator
    missile_queue: list[MissileEvent] = field(default_factory=list)
    tracker: MetricsTracker = field(default_factory=MetricsTracker)
    done: bool = False
    winner: str = "ongoing"
    termination_reason: str = ""
    last_round_reward_red: float = 0.0
    last_round_reward_blue: float = 0.0
    cache: StepCache | None = None
