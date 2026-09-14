from __future__ import annotations

import numpy as np

from skyarena2d.adapters.action_types import SkyArenaSideAction
from skyarena2d.opponents import RULES, RuleOpponentPool
from skyarena2d.rl.adapters.vector_env import SerialEnvRunner, SubprocessEnvRunner
from skyarena2d.rl.utils.config import load_mappo_config


def _no_fire_action(context) -> SkyArenaSideAction:
    n_agents = context.num_fighters
    alive = np.asarray(context.alive, dtype=bool)
    radar = np.zeros(n_agents, dtype=np.int32)
    radar[alive] = 1
    return SkyArenaSideAction(
        course=context.current_heading.copy(),
        radar_freq=radar,
        jammer_freq=np.zeros(n_agents, dtype=np.int32),
        fire_type=np.zeros(n_agents, dtype=np.int32),
        target_idx=np.full(n_agents, -1, dtype=np.int32),
    )


def _assert_observations_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_allclose(left[key], right[key], rtol=0.0, atol=0.0)


def test_subprocess_runner_matches_serial_seeded_trajectory() -> None:
    cfg = load_mappo_config("configs/mappo_skyarena_baseline_v2_smoke.yaml")
    cfg["train"]["num_envs"] = 2
    serial = SerialEnvRunner(cfg, num_envs=2)
    parallel = SubprocessEnvRunner(cfg, num_envs=2, num_workers=2)
    try:
        serial_obs = serial.reset_all()
        parallel_obs = parallel.reset_all()
        for left, right in zip(serial_obs, parallel_obs, strict=True):
            _assert_observations_equal(left, right)

        for _ in range(4):
            actions = [_no_fire_action(context) for context in serial.action_contexts]
            serial_results = serial.step_all(actions)
            parallel_results = parallel.step_all(actions)
            for serial_result, parallel_result in zip(
                serial_results, parallel_results, strict=True
            ):
                left_obs, left_reward, left_done, left_info = serial_result
                right_obs, right_reward, right_done, right_info = parallel_result
                _assert_observations_equal(left_obs, right_obs)
                assert left_reward == right_reward
                assert left_done == right_done
                assert left_info["winner"] == right_info["winner"]
                assert left_info["opponent_name"] == right_info["opponent_name"]
    finally:
        serial.close()
        parallel.close()


def test_parallel_eval_offsets_match_serial_seed_sequence() -> None:
    cfg = load_mappo_config("configs/mappo_skyarena_baseline_v2_smoke.yaml")
    cfg["env"]["opponent_pool"] = []
    serial = SerialEnvRunner(
        cfg,
        num_envs=1,
        seed_offsets=[9999],
        deterministic_reset=True,
    )
    parallel = SubprocessEnvRunner(
        cfg,
        num_envs=3,
        num_workers=3,
        seed_offsets=[9999, 10000, 10001],
        deterministic_reset=True,
    )
    try:
        serial_observations = [serial.reset_all()[0]]
        serial_observations.extend(serial.reset_at(0) for _ in range(2))
        parallel_observations = parallel.reset_all()

        assert serial.last_reset_seeds[0] == 10043
        assert parallel.last_reset_seeds == [10041, 10042, 10043]
        for left, right in zip(
            serial_observations, parallel_observations, strict=True
        ):
            _assert_observations_equal(left, right)
    finally:
        serial.close()
        parallel.close()


def test_subprocess_pool_results_are_recorded_centrally() -> None:
    parallel = object.__new__(SubprocessEnvRunner)
    parallel.num_envs = 2
    parallel._assignments = [[0, 1]]
    parallel.action_contexts = [None, None]
    parallel.opponent_pool = RuleOpponentPool(
        ["no_attack_rule", "fix_rule_v2"], RULES
    )
    fake_rows = [
        (
            index,
            ({}, 0.0, True, {"opponent_name": name, "winner": "red"}),
            object(),
        )
        for index, name in enumerate(parallel.opponent_pool.names)
    ]
    parallel._exchange = lambda _command, _payloads: fake_rows

    parallel.step_all([None, None])

    assert sum(
        record.games for record in parallel.opponent_pool.records.values()
    ) == 2
