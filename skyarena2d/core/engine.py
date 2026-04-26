from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..adapters.action_decoder import DecodedSideAction, decode_maca_side_action
from ..adapters.maca_compat import build_maca_raw_obs
from ..adapters.modern_obs_builder import build_modern_obs
from ..render.pixel_renderer import PixelRenderer
from .config import EnvConfig, FighterProfile, load_config
from .dynamics import apply_motion
from .jamming import JammingResult, compute_jammed_matrix
from .metrics import build_metrics_snapshot, update_tracker
from .passive_detection import compute_passive_detection
from .reward import compute_rewards
from .sensors import compute_visible_matrix
from .spawn import generate_spawn_positions
from .state import UNIT_DETECTOR, UNIT_FIGHTER, EnvState, LaunchRecord, StepCache, TeamState
from .termination import check_termination
from .weapons import WeaponStepResult, process_weapons


def _profile_value(profile: FighterProfile | None, key: str, fallback: float | int) -> float | int:
    if profile is None:
        return fallback
    value = getattr(profile, key)
    return fallback if value is None else value


class SkyArenaEngine:
    def __init__(
        self,
        config: EnvConfig | str | Path,
        render_mode: str | None = None,
    ) -> None:
        self.config = load_config(config) if isinstance(config, (str, Path)) else config
        self.render_mode = render_mode
        self.state: EnvState | None = None
        self._episode_counter = 0
        self._renderer: PixelRenderer | None = None
        self._last_obs: dict[str, Any] | None = None
        self._last_info: dict[str, Any] | None = None

    def _build_team(
        self,
        *,
        side: str,
        num_fighters: int,
        num_detectors: int,
        positions: np.ndarray,
        heading_deg: float | np.ndarray,
        profiles: list[FighterProfile],
    ) -> TeamState:
        total = num_fighters + num_detectors
        alive = np.ones((total,), dtype=bool)
        pos = positions.astype(np.float32).copy()
        heading_arr = np.asarray(heading_deg, dtype=np.float32)
        if heading_arr.ndim == 1 and heading_arr.shape[0] == total:
            heading = heading_arr.copy()
        else:
            heading = np.full((total,), float(heading_deg), dtype=np.float32)

        speed = np.full((total,), self.config.dynamics.default_detector_speed, dtype=np.float32)
        speed[:num_fighters] = self.config.dynamics.default_fighter_speed

        unit_type = np.full((total,), UNIT_DETECTOR, dtype=np.int32)
        unit_type[:num_fighters] = UNIT_FIGHTER

        radar_on = np.full((total,), self.config.radar.enabled, dtype=bool)
        radar_freq = np.full((total,), 1 if self.config.radar.enabled else 0, dtype=np.int32)

        jammer_on = np.zeros((total,), dtype=bool)
        jammer_freq = np.zeros((total,), dtype=np.int32)

        long_ammo = np.zeros((total,), dtype=np.int32)
        short_ammo = np.zeros((total,), dtype=np.int32)
        long_ammo[:num_fighters] = self.config.weapon.long_ammo
        short_ammo[:num_fighters] = self.config.weapon.short_ammo

        radar_range = np.full((total,), self.config.radar.detector_range, dtype=np.float32)
        radar_fov = np.full((total,), self.config.radar.detector_fov_deg, dtype=np.float32)
        radar_range[:num_fighters] = self.config.radar.fighter_range
        radar_fov[:num_fighters] = self.config.radar.fighter_fov_deg

        jammer_range = np.zeros((total,), dtype=np.float32)
        jammer_range[:num_fighters] = self.config.jamming.range

        long_range = np.zeros((total,), dtype=np.float32)
        short_range = np.zeros((total,), dtype=np.float32)
        long_range[:num_fighters] = self.config.weapon.long_range
        short_range[:num_fighters] = self.config.weapon.short_range

        long_hit_prob = np.zeros((total,), dtype=np.float32)
        short_hit_prob = np.zeros((total,), dtype=np.float32)
        long_hit_prob[:num_fighters] = self.config.weapon.long_hit_prob
        short_hit_prob[:num_fighters] = self.config.weapon.short_hit_prob

        for i in range(num_fighters):
            profile = profiles[i] if i < len(profiles) else None
            speed[i] = float(_profile_value(profile, "speed", speed[i]))
            radar_range[i] = float(_profile_value(profile, "radar_range", radar_range[i]))
            radar_fov[i] = float(_profile_value(profile, "radar_fov_deg", radar_fov[i]))
            jammer_range[i] = float(_profile_value(profile, "jammer_range", jammer_range[i]))
            long_range[i] = float(_profile_value(profile, "long_range", long_range[i]))
            short_range[i] = float(_profile_value(profile, "short_range", short_range[i]))
            long_hit_prob[i] = float(_profile_value(profile, "long_hit_prob", long_hit_prob[i]))
            short_hit_prob[i] = float(_profile_value(profile, "short_hit_prob", short_hit_prob[i]))
            long_ammo[i] = int(_profile_value(profile, "long_ammo", int(long_ammo[i])))
            short_ammo[i] = int(_profile_value(profile, "short_ammo", int(short_ammo[i])))

        last_action = np.zeros((total, 4), dtype=np.float32)
        last_reward = np.zeros((total,), dtype=np.float32)

        return TeamState(
            side="red" if side == "red" else "blue",
            num_fighters=num_fighters,
            num_detectors=num_detectors,
            alive=alive,
            pos=pos,
            heading=heading,
            speed=speed,
            unit_type=unit_type,
            radar_on=radar_on,
            radar_freq=radar_freq,
            jammer_on=jammer_on,
            jammer_freq=jammer_freq,
            long_ammo=long_ammo,
            short_ammo=short_ammo,
            radar_range=radar_range,
            radar_fov_deg=radar_fov,
            jammer_range=jammer_range,
            long_range=long_range,
            short_range=short_range,
            long_hit_prob=long_hit_prob,
            short_hit_prob=short_hit_prob,
            last_action=last_action,
            last_reward=last_reward,
        )

    def _zero_side_action(self, team: TeamState) -> dict[str, np.ndarray]:
        return {
            "fighter_action": np.zeros((team.num_fighters, 4), dtype=np.float32),
            "detector_action": np.zeros((team.num_detectors, 2), dtype=np.float32),
        }

    def _decode_actions(self, actions: dict[str, Any]) -> tuple[DecodedSideAction, DecodedSideAction]:
        assert self.state is not None
        red_input = actions.get("red", {}) if actions else {}
        blue_input = actions.get("blue", {}) if actions else {}

        red_decoded = decode_maca_side_action(
            self.state.red,
            red_input.get("fighter_action"),
            red_input.get("detector_action"),
            self.config.radar.freq_count,
        )
        blue_decoded = decode_maca_side_action(
            self.state.blue,
            blue_input.get("fighter_action"),
            blue_input.get("detector_action"),
            self.config.radar.freq_count,
        )
        return red_decoded, blue_decoded

    def _apply_side_action(self, team: TeamState, decoded: DecodedSideAction) -> None:
        team.heading = decoded.course.astype(np.float32)
        team.radar_freq = decoded.radar_freq.astype(np.int32)
        team.radar_on = team.radar_freq > 0

        team.jammer_freq[:] = 0
        if team.num_fighters > 0:
            team.jammer_freq[: team.num_fighters] = decoded.jammer_freq[: team.num_fighters]
        team.jammer_on = team.jammer_freq > 0

        team.last_action[:] = 0.0
        if team.num_fighters > 0:
            team.last_action[: team.num_fighters, :] = decoded.fighter_action
        if team.num_detectors > 0:
            start = team.num_fighters
            stop = team.total_units
            team.last_action[start:stop, :2] = decoded.detector_action

    def _collect_strike_list(
        self,
        launch_records: list[LaunchRecord],
        resolved_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        strike_list: list[dict[str, Any]] = []
        for rec in launch_records:
            strike_list.append(
                {
                    "attacker_side": rec.attacker_side,
                    "attacker_idx": rec.attacker_idx,
                    "target_side": rec.target_side,
                    "target_idx": rec.target_idx,
                    "missile_type": rec.missile_type,
                    "hit_prob": rec.hit_prob,
                    "valid": rec.valid,
                    "status": "launched" if rec.valid else "invalid",
                    "step": self.state.step_count if self.state else 0,
                }
            )

        for pending in self.state.missile_queue if self.state else []:
            strike_list.append(
                {
                    "attacker_side": pending.attacker_side,
                    "attacker_idx": pending.attacker_idx,
                    "target_side": pending.target_side,
                    "target_idx": pending.target_idx,
                    "missile_type": pending.missile_type,
                    "hit_prob": pending.hit_prob,
                    "status": "pending",
                    "launch_step": pending.launch_step,
                    "resolve_step": pending.resolve_step,
                }
            )

        for resolved in resolved_records:
            incoming = resolved.get("incoming", [])
            if not isinstance(incoming, list):
                continue
            for inc in incoming:
                if not isinstance(inc, dict):
                    continue
                strike_list.append(
                    {
                        "attacker_side": inc.get("attacker_side"),
                        "attacker_idx": inc.get("attacker_idx"),
                        "target_side": resolved.get("target_side"),
                        "target_idx": resolved.get("target_idx"),
                        "missile_type": inc.get("missile_type"),
                        "hit_prob": inc.get("hit_prob"),
                        "status": "resolved_hit"
                        if resolved.get("target_destroyed", False)
                        else "resolved_miss",
                    }
                )
        return strike_list

    def _build_observations(self) -> dict[str, Any]:
        assert self.state is not None
        assert self.state.cache is not None
        cache = self.state.cache

        red_raw = build_maca_raw_obs(
            own=self.state.red,
            enemy=self.state.blue,
            visible_matrix=cache.red_visible,
            distance_matrix=self._red_sensor.distance_matrix,
            bearing_matrix=self._red_sensor.relative_bearing_matrix,
            passive_lists=cache.red_passive,
            strike_list=cache.strike_list,
        )
        blue_raw = build_maca_raw_obs(
            own=self.state.blue,
            enemy=self.state.red,
            visible_matrix=cache.blue_visible,
            distance_matrix=self._blue_sensor.distance_matrix,
            bearing_matrix=self._blue_sensor.relative_bearing_matrix,
            passive_lists=cache.blue_passive,
            strike_list=cache.strike_list,
        )

        red_modern = build_modern_obs(
            self.state.red,
            self.state.blue,
            cache.red_visible,
            cache.red_fireable_long,
            cache.red_fireable_short,
            self.config.map.width,
            self.config.map.height,
            self.config.radar.freq_count,
        )
        blue_modern = build_modern_obs(
            self.state.blue,
            self.state.red,
            cache.blue_visible,
            cache.blue_fireable_long,
            cache.blue_fireable_short,
            self.config.map.width,
            self.config.map.height,
            self.config.radar.freq_count,
        )

        return {
            "red": {"raw": red_raw, "modern": red_modern},
            "blue": {"raw": blue_raw, "modern": blue_modern},
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = options
        rng = np.random.default_rng(seed)

        red_total = self.config.teams.red_fighters + self.config.teams.red_detectors
        blue_total = self.config.teams.blue_fighters + self.config.teams.blue_detectors
        red_pos, blue_pos = generate_spawn_positions(
            self.config,
            red_total,
            blue_total,
            rng,
            self._episode_counter,
            0,
        )

        red_heading: float | np.ndarray = 0.0
        blue_heading: float | np.ndarray = 180.0
        if self.config.spawn.spread and self.config.spawn.heading_spread > 0:
            hs = self.config.spawn.heading_spread
            red_heading = rng.uniform(-hs, hs, size=red_total).astype(np.float32)
            blue_heading = 180.0 + rng.uniform(-hs, hs, size=blue_total).astype(np.float32)

        red = self._build_team(
            side="red",
            num_fighters=self.config.teams.red_fighters,
            num_detectors=self.config.teams.red_detectors,
            positions=red_pos,
            heading_deg=red_heading,
            profiles=self.config.red_fighter_profiles,
        )
        blue = self._build_team(
            side="blue",
            num_fighters=self.config.teams.blue_fighters,
            num_detectors=self.config.teams.blue_detectors,
            positions=blue_pos,
            heading_deg=blue_heading,
            profiles=self.config.blue_fighter_profiles,
        )

        self.state = EnvState(
            red=red,
            blue=blue,
            step_count=0,
            episode_idx=self._episode_counter,
            rng=rng,
        )
        self._episode_counter += 1

        # Seed caches with no-jam baseline so reset returns valid obs.
        red_jam = JammingResult(
            jammed_matrix=np.zeros((red.total_units, blue.total_units), dtype=bool),
            jammed_detection_count=0,
            debug={},
        )
        blue_jam = JammingResult(
            jammed_matrix=np.zeros((blue.total_units, red.total_units), dtype=bool),
            jammed_detection_count=0,
            debug={},
        )
        self._red_sensor = compute_visible_matrix(red, blue, red_jam.jammed_matrix)
        self._blue_sensor = compute_visible_matrix(blue, red, blue_jam.jammed_matrix)
        red_passive, _ = compute_passive_detection(red, blue, self.config.passive_detection)
        blue_passive, _ = compute_passive_detection(blue, red, self.config.passive_detection)

        empty_weapons = WeaponStepResult(
            red_fireable_long=np.zeros((red.total_units, blue.total_units), dtype=bool),
            red_fireable_short=np.zeros((red.total_units, blue.total_units), dtype=bool),
            blue_fireable_long=np.zeros((blue.total_units, red.total_units), dtype=bool),
            blue_fireable_short=np.zeros((blue.total_units, red.total_units), dtype=bool),
            launch_records=[],
            resolved_records=[],
            red_valid_fire=np.zeros((red.total_units,), dtype=bool),
            red_invalid_fire=np.zeros((red.total_units,), dtype=bool),
            blue_valid_fire=np.zeros((blue.total_units,), dtype=bool),
            blue_invalid_fire=np.zeros((blue.total_units,), dtype=bool),
            killed_red=[],
            killed_blue=[],
            missiles_launched_long=0,
            missiles_launched_short=0,
            missiles_hit=0,
            missiles_missed=0,
        )
        strike_list = []
        self.state.cache = StepCache(
            red_visible=self._red_sensor.visible_matrix,
            blue_visible=self._blue_sensor.visible_matrix,
            red_jammed=red_jam.jammed_matrix,
            blue_jammed=blue_jam.jammed_matrix,
            red_fireable_long=empty_weapons.red_fireable_long,
            red_fireable_short=empty_weapons.red_fireable_short,
            blue_fireable_long=empty_weapons.blue_fireable_long,
            blue_fireable_short=empty_weapons.blue_fireable_short,
            red_passive=red_passive,
            blue_passive=blue_passive,
            launch_records=[],
            resolved_records=[],
            strike_list=strike_list,
            red_attempted_fire=np.zeros((red.total_units,), dtype=bool),
            blue_attempted_fire=np.zeros((blue.total_units,), dtype=bool),
            red_selected_long=np.zeros((red.total_units,), dtype=bool),
            red_selected_short=np.zeros((red.total_units,), dtype=bool),
            blue_selected_long=np.zeros((blue.total_units,), dtype=bool),
            blue_selected_short=np.zeros((blue.total_units,), dtype=bool),
            red_selected_target_idx=np.full((red.total_units,), -1, dtype=np.int32),
            blue_selected_target_idx=np.full((blue.total_units,), -1, dtype=np.int32),
            red_attempted_long_matrix=np.zeros((red.total_units, blue.total_units), dtype=bool),
            red_attempted_short_matrix=np.zeros((red.total_units, blue.total_units), dtype=bool),
            blue_attempted_long_matrix=np.zeros((blue.total_units, red.total_units), dtype=bool),
            blue_attempted_short_matrix=np.zeros((blue.total_units, red.total_units), dtype=bool),
            red_selected_long_matrix=np.zeros((red.total_units, blue.total_units), dtype=bool),
            red_selected_short_matrix=np.zeros((red.total_units, blue.total_units), dtype=bool),
            blue_selected_long_matrix=np.zeros((blue.total_units, red.total_units), dtype=bool),
            blue_selected_short_matrix=np.zeros((blue.total_units, red.total_units), dtype=bool),
            reward_components={},
        )

        obs = self._build_observations()
        info = {
            "winner": "ongoing",
            "reason": "",
            "metrics": build_metrics_snapshot(self.state, empty_weapons),
            "maca_reward": {
                "side1_detector_reward": [0.0] * red.num_detectors,
                "side1_fighter_reward": [0.0] * red.num_fighters,
                "side1_round_reward": 0.0,
                "side2_detector_reward": [0.0] * blue.num_detectors,
                "side2_fighter_reward": [0.0] * blue.num_fighters,
                "side2_round_reward": 0.0,
            },
        }
        self._last_obs = obs
        self._last_info = info
        return obs, info

    def step(
        self,
        actions: dict[str, dict[str, np.ndarray]],
    ) -> tuple[dict[str, Any], dict[str, Any], bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.state.done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        self.state.step_count += 1
        red_decoded, blue_decoded = self._decode_actions(actions)

        self._apply_side_action(self.state.red, red_decoded)
        self._apply_side_action(self.state.blue, blue_decoded)

        apply_motion(
            self.state.red,
            self.config.dynamics,
            self.config.map.width,
            self.config.map.height,
            self.config.dt,
        )
        apply_motion(
            self.state.blue,
            self.config.dynamics,
            self.config.map.width,
            self.config.map.height,
            self.config.dt,
        )

        red_jam = compute_jammed_matrix(
            self.state.red,
            self.state.blue,
            self.config.radar,
            self.config.jamming,
            self.state.rng,
        )
        blue_jam = compute_jammed_matrix(
            self.state.blue,
            self.state.red,
            self.config.radar,
            self.config.jamming,
            self.state.rng,
        )

        self._red_sensor = compute_visible_matrix(self.state.red, self.state.blue, red_jam.jammed_matrix)
        self._blue_sensor = compute_visible_matrix(self.state.blue, self.state.red, blue_jam.jammed_matrix)

        red_passive, passive_count_red = compute_passive_detection(
            self.state.red,
            self.state.blue,
            self.config.passive_detection,
        )
        blue_passive, passive_count_blue = compute_passive_detection(
            self.state.blue,
            self.state.red,
            self.config.passive_detection,
        )

        weapon_result = process_weapons(
            state=self.state,
            config=self.config,
            red_hit_targets=red_decoded.hit_target,
            blue_hit_targets=blue_decoded.hit_target,
            red_visible=self._red_sensor.visible_matrix,
            blue_visible=self._blue_sensor.visible_matrix,
            red_passive=red_passive,
            blue_passive=blue_passive,
        )

        self.state.red.kills += len(weapon_result.killed_blue)
        self.state.blue.losses += len(weapon_result.killed_blue)
        self.state.blue.kills += len(weapon_result.killed_red)
        self.state.red.losses += len(weapon_result.killed_red)

        update_tracker(
            state=self.state,
            weapon_result=weapon_result,
            red_visible=self._red_sensor.visible_matrix,
            blue_visible=self._blue_sensor.visible_matrix,
            red_jammed_count=red_jam.jammed_detection_count,
            blue_jammed_count=blue_jam.jammed_detection_count,
            passive_count_red=passive_count_red,
            passive_count_blue=passive_count_blue,
        )

        termination_result = check_termination(self.state, self.config)
        reward_output = compute_rewards(
            state=self.state,
            config=self.config,
            weapon_result=weapon_result,
            termination_result=termination_result,
        )

        self.state.red.last_reward = reward_output.red_unit_rewards.astype(np.float32)
        self.state.blue.last_reward = reward_output.blue_unit_rewards.astype(np.float32)
        self.state.done = termination_result.done
        self.state.winner = termination_result.winner
        self.state.termination_reason = termination_result.reason

        strike_list = self._collect_strike_list(weapon_result.launch_records, weapon_result.resolved_records)
        self.state.cache = StepCache(
            red_visible=self._red_sensor.visible_matrix,
            blue_visible=self._blue_sensor.visible_matrix,
            red_jammed=red_jam.jammed_matrix,
            blue_jammed=blue_jam.jammed_matrix,
            red_fireable_long=weapon_result.red_fireable_long,
            red_fireable_short=weapon_result.red_fireable_short,
            blue_fireable_long=weapon_result.blue_fireable_long,
            blue_fireable_short=weapon_result.blue_fireable_short,
            red_passive=red_passive,
            blue_passive=blue_passive,
            launch_records=weapon_result.launch_records,
            resolved_records=weapon_result.resolved_records,
            strike_list=strike_list,
            red_attempted_fire=weapon_result.red_attempted_fire,
            blue_attempted_fire=weapon_result.blue_attempted_fire,
            red_selected_long=weapon_result.red_selected_long,
            red_selected_short=weapon_result.red_selected_short,
            blue_selected_long=weapon_result.blue_selected_long,
            blue_selected_short=weapon_result.blue_selected_short,
            red_selected_target_idx=weapon_result.red_selected_target_idx,
            blue_selected_target_idx=weapon_result.blue_selected_target_idx,
            red_attempted_long_matrix=weapon_result.red_attempted_long_matrix,
            red_attempted_short_matrix=weapon_result.red_attempted_short_matrix,
            blue_attempted_long_matrix=weapon_result.blue_attempted_long_matrix,
            blue_attempted_short_matrix=weapon_result.blue_attempted_short_matrix,
            red_selected_long_matrix=weapon_result.red_selected_long_matrix,
            red_selected_short_matrix=weapon_result.red_selected_short_matrix,
            blue_selected_long_matrix=weapon_result.blue_selected_long_matrix,
            blue_selected_short_matrix=weapon_result.blue_selected_short_matrix,
            reward_components=reward_output.components,
        )

        metrics = build_metrics_snapshot(self.state, weapon_result)

        obs = self._build_observations()
        reward = {
            "red": float(reward_output.red_team_reward),
            "blue": float(reward_output.blue_team_reward),
            "red_unit_sum": float(np.sum(reward_output.red_unit_rewards)),
            "blue_unit_sum": float(np.sum(reward_output.blue_unit_rewards)),
            "red_unit": reward_output.red_unit_rewards.copy(),
            "blue_unit": reward_output.blue_unit_rewards.copy(),
            "maca": reward_output.maca_reward,
        }
        info = {
            "winner": termination_result.winner,
            "reason": termination_result.reason,
            "metrics": metrics,
            "reward_components": reward_output.components,
            "maca_reward": reward_output.maca_reward,
        }

        self._last_obs = obs
        self._last_info = info
        return obs, reward, termination_result.done, termination_result.truncated, info

    def get_last_obs(self) -> dict[str, Any] | None:
        return self._last_obs

    def get_last_info(self) -> dict[str, Any] | None:
        return self._last_info

    def get_state(self) -> EnvState:
        if self.state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return self.state

    def render(self, render_mode: str | None = None) -> np.ndarray | None:
        if self.state is None:
            return None

        mode = render_mode or self.render_mode
        if mode is None:
            return None

        if self._renderer is None:
            self._renderer = PixelRenderer(self.config)
        metrics = self._last_info.get("metrics") if self._last_info else None
        return self._renderer.render(self.state, mode=mode, metrics=metrics)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
