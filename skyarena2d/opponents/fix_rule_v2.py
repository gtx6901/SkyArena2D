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
        jammer_freq: int = 0,
        prefer_long_before_short: bool = True,
        long_range: float = 220.0,
        short_range: float = 120.0,
    ) -> None:
        super().__init__(seed)
        self.initial_push_by_side = initial_push_by_side
        self.random_hold_steps = random_hold_steps
        self.radar_freq = radar_freq
        self.jammer_freq = jammer_freq
        self.prefer_long = prefer_long_before_short
        self.long_range = long_range
        self.short_range = short_range
        self.first_contact_step: int | None = None
        self.agent_search_heading: dict[int, float] = {}
        self.agent_search_timer: dict[int, int] = {}
        # track last fired step per agent to avoid spamming
        self.agent_last_fired: dict[int, int] = {}
        self.fire_cooldown: int = 5  # steps between shots per agent

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.first_contact_step = None
        self.agent_search_heading.clear()
        self.agent_search_timer.clear()
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
            return {"fighter_action": fighter_action, "detector_action": detector_action}

        # Phase 2: post-contact, each agent decides independently
        max_enemy = side_obs["modern"]["enemies"].shape[1]
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

        return {"fighter_action": fighter_action, "detector_action": detector_action}
