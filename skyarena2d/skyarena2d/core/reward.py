from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EnvConfig
from .state import EnvState
from .termination import TerminationResult
from .weapons import WeaponStepResult


@dataclass(slots=True)
class RewardOutput:
    red_unit_rewards: np.ndarray
    blue_unit_rewards: np.ndarray
    maca_reward: dict[str, object]


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
) -> None:
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


def compute_rewards(
    *,
    state: EnvState,
    config: EnvConfig,
    weapon_result: WeaponStepResult,
    termination_result: TerminationResult,
) -> RewardOutput:
    red_rewards = np.zeros((state.red.total_units,), dtype=np.float32)
    blue_rewards = np.zeros((state.blue.total_units,), dtype=np.float32)

    red_rewards[weapon_result.red_valid_fire] += config.reward.valid_fire
    red_rewards[weapon_result.red_invalid_fire] += config.reward.invalid_fire
    blue_rewards[weapon_result.blue_valid_fire] += config.reward.valid_fire
    blue_rewards[weapon_result.blue_invalid_fire] += config.reward.invalid_fire

    _apply_resolution_rewards(
        state=state,
        config=config,
        weapon_result=weapon_result,
        red_rewards=red_rewards,
        blue_rewards=blue_rewards,
    )

    if config.reward.keep_alive_step != 0.0:
        red_rewards[state.red.alive] += config.reward.keep_alive_step
        blue_rewards[state.blue.alive] += config.reward.keep_alive_step

    red_round = 0.0
    blue_round = 0.0
    if termination_result.done:
        if termination_result.winner == "red":
            red_round = config.reward.win
            blue_round = config.reward.lose
        elif termination_result.winner == "blue":
            red_round = config.reward.lose
            blue_round = config.reward.win
        else:
            red_round = config.reward.draw
            blue_round = config.reward.draw

        red_rewards += red_round
        blue_rewards += blue_round

    state.last_round_reward_red = red_round
    state.last_round_reward_blue = blue_round

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
        maca_reward=maca_reward,
    )
