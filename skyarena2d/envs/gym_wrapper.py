from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.config import EnvConfig, load_config
from ..core.engine import SkyArenaEngine


class SkyArenaGymWrapper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: str | EnvConfig,
        blue_policy: Callable[[dict[str, Any]], dict[str, np.ndarray]] | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = load_config(config) if isinstance(config, str) else config
        self.engine = SkyArenaEngine(self.config, render_mode=render_mode)
        self.blue_policy = blue_policy

        max_enemy = self.config.teams.blue_fighters + self.config.teams.blue_detectors
        self.action_space = spaces.Dict(
            {
                "fighter_action": spaces.Box(
                    low=np.array([0, 0, 0, 0], dtype=np.float32),
                    high=np.array([359, self.config.radar.freq_count + 1, self.config.radar.freq_count + 1, 2 * max_enemy], dtype=np.float32),
                    shape=(self.config.teams.red_fighters, 4),
                    dtype=np.float32,
                ),
                "detector_action": spaces.Box(
                    low=np.array([0, 0], dtype=np.float32),
                    high=np.array([359, self.config.radar.freq_count], dtype=np.float32),
                    shape=(self.config.teams.red_detectors, 2),
                    dtype=np.float32,
                ),
            }
        )

        obs_dim = self._obs_dim("red")
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self._last_obs: dict[str, Any] | None = None

    def _obs_dim(self, side: str) -> int:
        if side == "red":
            own_n = self.config.teams.red_fighters + self.config.teams.red_detectors
            enemy_n = self.config.teams.blue_fighters + self.config.teams.blue_detectors
        else:
            own_n = self.config.teams.blue_fighters + self.config.teams.blue_detectors
            enemy_n = self.config.teams.red_fighters + self.config.teams.red_detectors
        return 14 + max(0, own_n - 1) * 10 + enemy_n * 13 + max(0, own_n - 1) + enemy_n + 1 + 12

    def _flatten_side_obs(self, side_modern: dict[str, Any]) -> np.ndarray:
        if side_modern["self"].shape[0] == 0:
            return np.zeros((self.observation_space.shape[0],), dtype=np.float32)
        idx = 0
        self_row = side_modern["self"][idx].reshape(-1)
        allies = side_modern["allies"][idx].reshape(-1)
        enemies = side_modern["enemies"][idx].reshape(-1)
        ally_mask = side_modern["masks"]["ally"][idx].astype(np.float32)
        enemy_mask = side_modern["masks"]["enemy"][idx].astype(np.float32)
        self_alive = np.array([float(side_modern["masks"]["self_alive"][idx])], dtype=np.float32)
        global_state = side_modern["global_state"].reshape(-1)
        return np.concatenate([self_row, allies, enemies, ally_mask, enemy_mask, self_alive, global_state]).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.engine.reset(seed=seed, options=options)
        self._last_obs = obs
        return self._flatten_side_obs(obs["red"]["modern"]), info

    def step(
        self,
        action: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._last_obs is None:
            raise RuntimeError("Call reset before step")

        if self.blue_policy is None:
            blue_action = {
                "fighter_action": np.zeros((self.config.teams.blue_fighters, 4), dtype=np.float32),
                "detector_action": np.zeros((self.config.teams.blue_detectors, 2), dtype=np.float32),
            }
        else:
            blue_action = self.blue_policy(self._last_obs["blue"])

        obs, reward, done, truncated, info = self.engine.step(
            {
                "red": {
                    "fighter_action": np.asarray(action.get("fighter_action", np.zeros((self.config.teams.red_fighters, 4))), dtype=np.float32),
                    "detector_action": np.asarray(action.get("detector_action", np.zeros((self.config.teams.red_detectors, 2))), dtype=np.float32),
                },
                "blue": {
                    "fighter_action": np.asarray(blue_action.get("fighter_action", np.zeros((self.config.teams.blue_fighters, 4))), dtype=np.float32),
                    "detector_action": np.asarray(blue_action.get("detector_action", np.zeros((self.config.teams.blue_detectors, 2))), dtype=np.float32),
                },
            }
        )
        self._last_obs = obs
        return self._flatten_side_obs(obs["red"]["modern"]), float(reward["red"]), done, truncated, info

    def render(self) -> np.ndarray | None:
        return self.engine.render()

    def close(self) -> None:
        self.engine.close()
