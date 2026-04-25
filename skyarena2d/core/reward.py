from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import EnvConfig
from .state import EnvState
from .termination import TerminationResult
from .weapons import WeaponStepResult


@dataclass(slots=True)
class RewardOutput:
    red_unit_rewards: np.ndarray
    blue_unit_rewards: np.ndarray
    red_team_reward: float
    blue_team_reward: float
    components: dict[str, Any] = field(default_factory=dict)
    # legacy compat
    maca_reward: dict[str, object] = field(default_factory=dict)


def _unit_kill_reward(is_fighter: bool, config: EnvConfig) -> float:
    return config.reward.kill_fighter if is_fighter else config.reward.kill_detector


def _unit_loss_reward(is_fighter: bool, config: EnvConfig) -> float:
    return config.reward.loss_fighter if is_fighter else config.reward.loss_detector


def _apply_resolution_rewards(
    *,
    state: EnvState,
    config: EnvConfig,
    weapon_result: WeaponStepResult,
    red_rewards: np.ndarray,
    blue_rewards: np.ndarray,
) -> dict[str, float]:
    red_kill_total = 0.0
    blue_kill_total = 0.0
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
            attacker_team = state.red
            attacker_rewards = red_rewards
            target_rewards = blue_rewards
        else:
            target_team = state.red
            attacker_team = state.blue
            attacker_rewards = blue_rewards
            target_rewards = red_rewards

        is_fighter = target_team.unit_type[target_idx] == 0
        kill_reward = _unit_kill_reward(bool(is_fighter), config)
        loss_reward = _unit_loss_reward(bool(is_fighter), config)

        target_rewards[target_idx] += loss_reward
        if target_side == "blue":
            blue_kill_total += loss_reward
        else:
            red_kill_total += loss_reward

        contributors: list[int] = []
        for item in incoming:
            if not isinstance(item, dict):
                continue
            contributor = int(item.get("attacker_idx", -1))
            if 0 <= contributor < attacker_team.total_units and contributor not in contributors:
                contributors.append(contributor)

        if contributors:
            share = kill_reward / len(contributors)
            attacker_rewards[np.array(contributors, dtype=np.int64)] += share
            if target_side == "blue":
                red_kill_total += kill_reward
            else:
                blue_kill_total += kill_reward

    return {"red_kill_loss": red_kill_total, "blue_kill_loss": blue_kill_total}


def compute_rewards(
    *,
    state: EnvState,
    config: EnvConfig,
    weapon_result: WeaponStepResult,
    termination_result: TerminationResult,
) -> RewardOutput:
    """Modular reward computation. Modules controlled by config.reward_modules if present."""
    red_rewards = np.zeros((state.red.total_units,), dtype=np.float32)
    blue_rewards = np.zeros((state.blue.total_units,), dtype=np.float32)
    components: dict[str, Any] = {}

    # Determine which modules are enabled (default: all legacy modules on)
    mods = getattr(config, "reward_modules", None) or {}

    def _mod_enabled(name: str, default: bool = True) -> bool:
        if not mods:
            return default
        m = mods.get(name, {})
        if isinstance(m, dict):
            return bool(m.get("enabled", default))
        return bool(m)

    def _mod_weight(name: str, key: str, fallback: float) -> float:
        if not mods:
            return fallback
        m = mods.get(name, {})
        if isinstance(m, dict):
            return float(m.get(key, fallback))
        return fallback

    # --- kill_loss module ---
    if _mod_enabled("kill_loss"):
        kl = _apply_resolution_rewards(
            state=state,
            config=config,
            weapon_result=weapon_result,
            red_rewards=red_rewards,
            blue_rewards=blue_rewards,
        )
        components["kill_loss"] = kl

    # --- valid_fire / invalid_fire module ---
    if _mod_enabled("valid_fire"):
        vf_coef = _mod_weight("valid_fire", "coef", config.reward.valid_fire)
        only_opp = _mod_weight("valid_fire", "only_when_opportunity", 0.0)
        if only_opp and weapon_result.red_fireable_long.size > 0:
            red_has_opp = np.any(weapon_result.red_fireable_long | weapon_result.red_fireable_short, axis=1)
            blue_has_opp = np.any(weapon_result.blue_fireable_long | weapon_result.blue_fireable_short, axis=1)
            red_rewards[weapon_result.red_valid_fire & red_has_opp] += vf_coef
            blue_rewards[weapon_result.blue_valid_fire & blue_has_opp] += vf_coef
        else:
            red_rewards[weapon_result.red_valid_fire] += vf_coef
            blue_rewards[weapon_result.blue_valid_fire] += vf_coef
        components["valid_fire"] = {
            "red": float(np.sum(red_rewards[weapon_result.red_valid_fire])),
            "blue": float(np.sum(blue_rewards[weapon_result.blue_valid_fire])),
        }

    if _mod_enabled("invalid_fire"):
        inv_penalty = _mod_weight("invalid_fire", "penalty", config.reward.invalid_fire)
        red_rewards[weapon_result.red_invalid_fire] += inv_penalty
        blue_rewards[weapon_result.blue_invalid_fire] += inv_penalty
        components["invalid_fire"] = {
            "red": float(np.sum(red_rewards[weapon_result.red_invalid_fire])),
            "blue": float(np.sum(blue_rewards[weapon_result.blue_invalid_fire])),
        }

    # --- fire_execution module: reward for firing when opportunity exists ---
    if _mod_enabled("fire_execution", default=False):
        fe_coef = _mod_weight("fire_execution", "coef", 0.01)
        if weapon_result.red_fireable_long.size > 0:
            red_has_opp = np.any(weapon_result.red_fireable_long | weapon_result.red_fireable_short, axis=1)
            blue_has_opp = np.any(weapon_result.blue_fireable_long | weapon_result.blue_fireable_short, axis=1)
            red_rewards[weapon_result.red_valid_fire & red_has_opp] += fe_coef
            blue_rewards[weapon_result.blue_valid_fire & blue_has_opp] += fe_coef
        components["fire_execution"] = {"coef": fe_coef}

    # --- selected_exchange module ---
    if _mod_enabled("selected_exchange", default=False):
        se_coef = _mod_weight("selected_exchange", "coef", 0.05)
        # reward agents that selected a target with positive expected exchange
        if weapon_result.red_selected_target_idx.size > 0:
            for i, t in enumerate(weapon_result.red_selected_target_idx):
                if int(t) >= 0 and i < state.red.num_fighters:
                    if bool(weapon_result.red_selected_long[i]):
                        p = float(state.red.long_hit_prob[i])
                    elif bool(weapon_result.red_selected_short[i]):
                        p = float(state.red.short_hit_prob[i])
                    else:
                        p = 0.0
                    red_rewards[i] += se_coef * p
        components["selected_exchange"] = {"coef": se_coef}

    # --- keep_alive_step ---
    if config.reward.keep_alive_step != 0.0:
        red_rewards[state.red.alive] += config.reward.keep_alive_step
        blue_rewards[state.blue.alive] += config.reward.keep_alive_step

    # --- win_loss module (terminal only, no double-counting) ---
    red_round = 0.0
    blue_round = 0.0
    if _mod_enabled("win_loss") and termination_result.done:
        win_r = _mod_weight("win_loss", "win", config.reward.win)
        lose_r = _mod_weight("win_loss", "lose", config.reward.lose)
        draw_r = _mod_weight("win_loss", "draw", config.reward.draw)
        if termination_result.winner == "red":
            red_round = win_r
            blue_round = lose_r
        elif termination_result.winner == "blue":
            red_round = lose_r
            blue_round = win_r
        else:
            red_round = draw_r
            blue_round = draw_r
        red_rewards += red_round
        blue_rewards += blue_round
        components["win_loss"] = {"red": red_round, "blue": blue_round}

    state.last_round_reward_red = red_round
    state.last_round_reward_blue = blue_round

    red_team = float(np.mean(red_rewards[state.red.alive]) if np.any(state.red.alive) else np.mean(red_rewards))
    blue_team = float(np.mean(blue_rewards[state.blue.alive]) if np.any(state.blue.alive) else np.mean(blue_rewards))

    maca_reward = {
        "side1_detector_reward": red_rewards[state.red.detector_indices].tolist(),
        "side1_fighter_reward": red_rewards[state.red.fighter_indices].tolist(),
        "side1_round_reward": float(red_round),
        "side2_detector_reward": blue_rewards[state.blue.detector_indices].tolist(),
        "side2_fighter_reward": blue_rewards[state.blue.fighter_indices].tolist(),
        "side2_round_reward": float(blue_round),
    }

    return RewardOutput(
        red_unit_rewards=red_rewards,
        blue_unit_rewards=blue_rewards,
        red_team_reward=red_team,
        blue_team_reward=blue_team,
        components=components,
        maca_reward=maca_reward,
    )
