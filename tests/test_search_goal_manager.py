from __future__ import annotations

import numpy as np

from skyarena2d.rl.algo.search_goal_manager import TeamSearchPlanner, TeamSearchPlannerConfig


def _obs() -> dict[str, np.ndarray]:
    return {
        "self_features": np.array([[[0.1] + [0.0] * 19, [0.2] + [0.0] * 19]], dtype=np.float32),
        "alive_mask": np.ones((1, 2), dtype=np.float32),
        "has_active_contact": np.zeros((1, 2), dtype=np.float32),
    }


def test_search_goal_refresh_executes_raw_action() -> None:
    mgr = TeamSearchPlanner(TeamSearchPlannerConfig(num_envs=1, num_agents=2, map_size_x=100.0, map_size_y=100.0))
    payload = mgr.apply(
        raw_goal_action=np.array([[3, 4]], dtype=np.int64),
        refresh_mask=np.array([[True, True]], dtype=bool),
        obs_batch=_obs(),
    )

    assert payload["executed_search_goal_action"].tolist() == [[3, 4]]
    assert payload["current_search_goal_id"].tolist() == [[4, 5]]


def test_search_goal_hold_keeps_existing_goal_without_refresh() -> None:
    mgr = TeamSearchPlanner(TeamSearchPlannerConfig(num_envs=1, num_agents=1, map_size_x=100.0, map_size_y=100.0))
    obs = {
        "self_features": np.zeros((1, 1, 20), dtype=np.float32),
        "alive_mask": np.ones((1, 1), dtype=np.float32),
        "has_active_contact": np.zeros((1, 1), dtype=np.float32),
    }
    mgr.apply(
        raw_goal_action=np.array([[7]], dtype=np.int64),
        refresh_mask=np.array([[True]], dtype=bool),
        obs_batch=obs,
    )
    payload = mgr.apply(
        raw_goal_action=np.array([[2]], dtype=np.int64),
        refresh_mask=np.array([[False]], dtype=bool),
        obs_batch=obs,
    )

    assert payload["executed_search_goal_action"].tolist() == [[7]]
