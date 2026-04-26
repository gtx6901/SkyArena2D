from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseRuleOpponent


class FixRuleV2Opponent(BaseRuleOpponent):
    """Improved fix_rule with initial push phase and per-agent independent decisions.

    Phase 1 (initial push, before first contact):
        Red: heading = 0 (east, toward blue)
        Blue: heading = 180 (west, toward red)
        First contact = any alive fighter sees any enemy in r_visible_list.

    Phase 2 (post-contact, each agent independently):
        If visible enemies exist:
            - Select nearest visible enemy
            - Set heading = absolute bearing to that enemy
            - Fire long if prefer_long and dist <= long_range
            - Fire short if dist <= short_range (and not prefer_long or out of long range)
        If no visible enemies:
            - Random search direction, held for random_hold_steps
            - Each agent independently resamples when timer expires
    """

    def __init__(
        self,
        seed: int | None = None,
        initial_push_by_side: bool = True,
        random_hold_steps: int = 40,
        radar_freq: int = 1,
        jammer_freq: int = 1,
        prefer_long_before_short: bool = True,
        long_range: float = 220.0,
        short_range: float = 120.0,
        use_jammer_strategy: bool = True,
        jammer_initial_silent: bool = True,
        jammer_on_after_contact: bool = True,
        jammer_range: float = 320.0,
        jammer_memory_steps: int = 10,
        match_target_radar_freq: bool = True,
        max_jammers_per_side: int = 3,
    ) -> None:
        super().__init__(seed)
        self.initial_push_by_side = initial_push_by_side
        self.random_hold_steps = random_hold_steps
        self.radar_freq = radar_freq
        self.jammer_freq = jammer_freq
        self.prefer_long = prefer_long_before_short
        self.long_range = long_range
        self.short_range = short_range
        self.use_jammer_strategy = use_jammer_strategy
        self.jammer_initial_silent = jammer_initial_silent
        self.jammer_on_after_contact = jammer_on_after_contact
        self.jammer_range = jammer_range
        self.jammer_memory_steps = jammer_memory_steps
        self.match_target_radar_freq = match_target_radar_freq
        self.max_jammers_per_side = max_jammers_per_side
        self.first_contact_step: int | None = None
        self.agent_search_heading: dict[int, float] = {}
        self.agent_search_timer: dict[int, int] = {}
        self.agent_last_jammer_contact: dict[int, int] = {}
        # track last fired step per agent to avoid spamming
        self.agent_last_fired: dict[int, int] = {}
        self.fire_cooldown: int = 5  # steps between shots per agent

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.first_contact_step = None
        self.agent_search_heading.clear()
        self.agent_search_timer.clear()
        self.agent_last_jammer_contact.clear()
        self.agent_last_fired.clear()

    def act(self, side_obs: dict[str, Any], side: str, step_count: int) -> dict[str, np.ndarray]:
        fighter_action, detector_action = self._base_actions(
            side_obs,
            course_value=0.0,
            radar_freq=self.radar_freq,
            jammer_freq=self.jammer_freq,
        )

        fighters = side_obs["raw"]["fighter_obs_list"]
        if len(fighters) == 0:
            return {"fighter_action": fighter_action, "detector_action": detector_action}

        # Detect first contact: any alive fighter sees any enemy
        has_any_contact = any(
            len(f.get("r_visible_list", [])) > 0
            for f in fighters
            if f.get("alive", False)
        )
        if has_any_contact and self.first_contact_step is None:
            self.first_contact_step = step_count

        # Phase 1: initial push before first contact
        if self.first_contact_step is None and self.initial_push_by_side:
            base_heading = 0.0 if side == "red" else 180.0
            fighter_action[:, 0] = base_heading
            if self.use_jammer_strategy and self.jammer_initial_silent:
                fighter_action[:, 2] = 0
            return {"fighter_action": fighter_action, "detector_action": detector_action}

        # Phase 2: post-contact, each agent decides independently
        max_enemy = side_obs["modern"]["enemies"].shape[1]
        jammer_requests: list[tuple[int, int, float]] = []
        for i, fighter in enumerate(fighters):
            if not fighter.get("alive", False):
                continue

            visible = fighter.get("r_visible_list", [])
            own_heading = float(fighter.get("course", 0.0))

            if visible:
                # Select nearest visible enemy
                nearest = min(visible, key=lambda x: float(x.get("distance", 1e9)))
                target_id = int(nearest["id"])
                dist = float(nearest.get("distance", 0.0))

                # direction is relative bearing (-180 to 180); convert to absolute heading
                direction = float(nearest.get("direction", 0.0))
                abs_bearing = (own_heading + direction) % 360.0
                fighter_action[i, 0] = abs_bearing

                # Reset search state since we have a target
                self.agent_search_timer[i] = 0
                jammer_freq = self._jammer_freq_for_target(nearest)
                if self._should_jam_visible_target(dist):
                    self.agent_last_jammer_contact[i] = step_count
                    jammer_requests.append((i, jammer_freq, dist))

                # Fire decision: check ammo + distance, with cooldown
                long_ammo = int(fighter.get("l_missile_left", 0))
                short_ammo = int(fighter.get("s_missile_left", 0))
                last_fired = self.agent_last_fired.get(i, -999)
                on_cooldown = (step_count - last_fired) < self.fire_cooldown
                if on_cooldown:
                    fighter_action[i, 3] = 0.0
                elif self.prefer_long and dist <= self.long_range and long_ammo > 0:
                    fighter_action[i, 3] = float(target_id)
                    self.agent_last_fired[i] = step_count
                elif dist <= self.short_range and short_ammo > 0:
                    fighter_action[i, 3] = float(max_enemy + target_id)
                    self.agent_last_fired[i] = step_count
                elif dist <= self.long_range and long_ammo > 0:
                    fighter_action[i, 3] = float(target_id)
                    self.agent_last_fired[i] = step_count
                else:
                    fighter_action[i, 3] = 0.0
            else:
                # Search mode: hold random heading for random_hold_steps
                timer = self.agent_search_timer.get(i, self.random_hold_steps)
                if i not in self.agent_search_heading or timer >= self.random_hold_steps:
                    self.agent_search_heading[i] = float(self.rng.uniform(0.0, 360.0))
                    self.agent_search_timer[i] = 0
                else:
                    self.agent_search_timer[i] = timer + 1

                fighter_action[i, 0] = self.agent_search_heading[i]
                fighter_action[i, 3] = 0.0  # no fire
                if self._should_hold_jammer(i, step_count):
                    jammer_requests.append((i, self.jammer_freq, self.jammer_range))

        if self.use_jammer_strategy:
            fighter_action[:, 2] = 0
            for i, freq, _dist in self._select_jammer_requests(jammer_requests):
                fighter_action[i, 2] = freq

        return {"fighter_action": fighter_action, "detector_action": detector_action}

    def _jammer_freq_for_target(self, target: dict[str, Any]) -> int:
        if self.match_target_radar_freq:
            for key in ("r_fp", "radar_freq", "r_fre_point"):
                value = int(target.get(key, 0) or 0)
                if value > 0:
                    return value
        return int(max(self.jammer_freq, 1))

    def _should_jam_visible_target(self, distance: float) -> bool:
        if not self.use_jammer_strategy or not self.jammer_on_after_contact:
            return False
        return float(distance) <= float(max(self.jammer_range, 0.0))

    def _should_hold_jammer(self, agent_idx: int, step_count: int) -> bool:
        if not self.use_jammer_strategy or not self.jammer_on_after_contact:
            return False
        last_contact = self.agent_last_jammer_contact.get(agent_idx)
        if last_contact is None:
            return False
        return (step_count - last_contact) <= max(int(self.jammer_memory_steps), 0)

    def _select_jammer_requests(self, requests: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        if not requests:
            return []
        max_jammers = max(int(self.max_jammers_per_side), 0)
        if max_jammers == 0:
            return []
        unique_by_agent: dict[int, tuple[int, int, float]] = {}
        for agent_idx, freq, distance in requests:
            if freq <= 0:
                continue
            prev = unique_by_agent.get(agent_idx)
            if prev is None or distance < prev[2]:
                unique_by_agent[agent_idx] = (agent_idx, int(freq), float(distance))
        ordered = sorted(unique_by_agent.values(), key=lambda row: row[2])
        return ordered[:max_jammers]
