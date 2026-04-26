from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class MapConfig(BaseModel):
    width: float = 2000.0
    height: float = 1200.0


class TeamConfig(BaseModel):
    red_fighters: int = 10
    blue_fighters: int = 10
    red_detectors: int = 0
    blue_detectors: int = 0


class FighterProfile(BaseModel):
    speed: float | None = None
    radar_range: float | None = None
    radar_fov_deg: float | None = None
    jammer_range: float | None = None
    long_range: float | None = None
    short_range: float | None = None
    long_hit_prob: float | None = None
    short_hit_prob: float | None = None
    long_ammo: int | None = None
    short_ammo: int | None = None


class DynamicsConfig(BaseModel):
    turn_mode: Literal["instant", "limited_turn_rate"] = "instant"
    boundary_mode: Literal["clamp", "bounce", "kill", "penalty"] = "clamp"
    default_fighter_speed: float = 2.0
    default_detector_speed: float = 1.5
    max_turn_rate_deg: float = 10.0
    boundary_penalty: float = 0.1


class CurriculumStage(BaseModel):
    until_episode: int = 0
    mode: Literal["fixed_scaled", "random_edge", "symmetric_random"] = "fixed_scaled"
    red_x_ratio: float | None = None
    blue_x_ratio: float | None = None
    y_min_ratio: float | None = None
    y_max_ratio: float | None = None
    jitter: float | None = None


class SpawnConfig(BaseModel):
    mode: Literal["fixed_scaled", "random_edge", "symmetric_random", "curriculum"] = "fixed_scaled"
    red_x_ratio: float = 0.18
    blue_x_ratio: float = 0.82
    y_min_ratio: float = 0.12
    y_max_ratio: float = 0.88
    jitter: float = 10.0
    edge_margin_ratio: float = 0.05
    min_y_gap_ratio: float = 0.02
    curriculum: list[CurriculumStage] = Field(default_factory=list)
    spread: bool = False
    x_spread: float = 120.0
    heading_spread: float = 8.0


class RadarConfig(BaseModel):
    enabled: bool = True
    freq_count: int = 10
    fighter_range: float = 260.0
    fighter_fov_deg: float = 120.0
    detector_range: float = 360.0
    detector_fov_deg: float = 360.0


class JammingConfig(BaseModel):
    enabled: bool = True
    mode: Literal["deterministic", "probabilistic"] = "deterministic"
    range: float = 300.0
    spot_block_prob: float = 1.0
    barrage_block_prob: float = 0.4


class PassiveDetectionConfig(BaseModel):
    enabled: bool = True
    detect_radar_on: bool = True
    detect_jammer_on: bool = True
    range: float = 500.0


class WeaponConfig(BaseModel):
    long_range: float = 220.0
    short_range: float = 120.0
    long_hit_prob: float = 0.35
    short_hit_prob: float = 0.60
    long_ammo: int = 2
    short_ammo: int = 4
    hit_prob_enable: bool = True
    attack_effect_delay: int = 0
    simultaneous_resolution: bool = True
    allow_passive_fire: bool = False


class RewardConfig(BaseModel):
    valid_fire: float = 0.02
    invalid_fire: float = -0.05
    kill_fighter: float = 1.0
    kill_detector: float = 0.5
    loss_fighter: float = -1.0
    loss_detector: float = -0.5
    win: float = 5.0
    lose: float = -5.0
    draw: float = 0.0
    keep_alive_step: float = 0.0


class RenderConfig(BaseModel):
    width: int = 1200
    height: int = 720
    pixel_style: bool = True
    debug_overlay: bool = True
    show_radar: bool = True
    show_jamming: bool = True
    show_visible_edges: bool = False
    show_fireable_edges: bool = True
    show_target_allocations: bool = True


class EnvConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    map: MapConfig = Field(default_factory=MapConfig)
    teams: TeamConfig = Field(default_factory=TeamConfig)
    max_steps: int = 1500
    dt: float = 1.0
    dynamics: DynamicsConfig = Field(default_factory=DynamicsConfig)
    spawn: SpawnConfig = Field(default_factory=SpawnConfig)
    radar: RadarConfig = Field(default_factory=RadarConfig)
    jamming: JammingConfig = Field(default_factory=JammingConfig)
    passive_detection: PassiveDetectionConfig = Field(default_factory=PassiveDetectionConfig)
    weapon: WeaponConfig = Field(default_factory=WeaponConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    # reward_modules: optional dict controlling which reward components are active
    reward_modules: dict = Field(default_factory=dict)
    render: RenderConfig = Field(default_factory=RenderConfig)
    red_fighter_profiles: list[FighterProfile] = Field(default_factory=list)
    blue_fighter_profiles: list[FighterProfile] = Field(default_factory=list)


def load_config(path: str | Path) -> EnvConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return EnvConfig.model_validate(data)
