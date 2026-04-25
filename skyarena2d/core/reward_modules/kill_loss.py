from __future__ import annotations

import numpy as np

from .base import RewardModule


class KillLossModule(RewardModule):
    """Kill/loss rewards based on resolved_records. Kill reward shared among contributors."""

    @property
    def name(self) -> str:
        return "kill_loss"

    def compute(self, state, config, weapon_result, termination_result) -> tuple[np.ndarray, np.ndarray, dict]:
        red_delta = np.zeros((state.red.total_units,), dtype=np.float32)
        blue_delta = np.zeros((state.blue.total_units,), dtype=np.float32)
        red_total = 0.0
        blue_total = 0.0

        for rec in weapon_result.resolved_records:
            if not bool(rec.get("target_destroyed", False)):
                continue

            target_side = str(rec["target_side"])
            target_idx = int(rec["target_idx"])
            incoming = rec.get("incoming", [])
            if not isinstance(incoming, list):
                incoming = []

            if target_side == "blue":
                target_team = state.blue
                attacker_rewards = red_delta
                target_rewards = blue_delta
                is_red_kill = True
            else:
                target_team = state.red
                attacker_rewards = blue_delta
                target_rewards = red_delta
                is_red_kill = False

            is_fighter = target_team.unit_type[target_idx] == 0
            kill_reward = config.reward.kill_fighter if is_fighter else config.reward.kill_detector
            loss_reward = config.reward.loss_fighter if is_fighter else config.reward.loss_detector

            target_rewards[target_idx] += loss_reward
            if is_red_kill:
                blue_total += loss_reward
            else:
                red_total += loss_reward

            contributors: list[int] = []
            for item in incoming:
                if not isinstance(item, dict):
                    continue
                contributor = int(item.get("attacker_idx", -1))
                if 0 <= contributor < attacker_team.total_units and contributor not in contributors:
                    contributors.append(contributor)
                    attacker_team = state.red if is_red_kill else state.blue

            if contributors:
                share = kill_reward / len(contributors)
                attacker_rewards[np.array(contributors, dtype=np.int64)] += share
                if is_red_kill:
                    red_total += kill_reward
                else:
                    blue_total += kill_reward

        return red_delta, blue_delta, {"red": red_total, "blue": blue_total}
