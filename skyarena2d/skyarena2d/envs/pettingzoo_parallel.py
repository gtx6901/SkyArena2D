from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from ..core.config import EnvConfig, load_config
from ..core.engine import SkyArenaEngine


def _flatten_agent_obs(side_obs: dict[str, Any], agent_idx: int) -> np.ndarray:
    self_row = side_obs["self"][agent_idx].astype(np.float32)
    allies = side_obs["allies"][agent_idx].reshape(-1).astype(np.float32)
    enemies = side_obs["enemies"][agent_idx].reshape(-1).astype(np.float32)
    ally_mask = side_obs["masks"]["ally"][agent_idx].astype(np.float32)
    enemy_mask = side_obs["masks"]["enemy"][agent_idx].astype(np.float32)
    self_alive = np.array([float(side_obs["masks"]["self_alive"][agent_idx])], dtype=np.float32)
    global_state = side_obs["global_state"].astype(np.float32)
    return np.concatenate([self_row, allies, enemies, ally_mask, enemy_mask, self_alive, global_state])


class SkyArenaParallelEnv(ParallelEnv):
    metadata = {"name": "SkyArenaParallelEnv", "render_modes": ["human", "rgb_array"]}

    def __init__(self, config: str | EnvConfig, render_mode: str | None = None) -> None:
        self.config = load_config(config) if isinstance(config, str) else config
        self.render_mode = render_mode
        self.engine = SkyArenaEngine(self.config, render_mode=render_mode)

        self._agent_specs: dict[str, tuple[str, int, str]] = {}
        self.possible_agents: list[str] = []
        self._build_agents()
        self.agents = self.possible_agents[:]

        self._obs_spaces: dict[str, spaces.Box] = {}
        self._act_spaces: dict[str, spaces.Space] = {}
        self._build_spaces()

    def _build_agents(self) -> None:
        self.possible_agents.clear()
        self._agent_specs.clear()

        for i in range(self.config.teams.red_fighters):
            name = f"red_fighter_{i}"
            self.possible_agents.append(name)
            self._agent_specs[name] = ("red", i, "fighter")
        for i in range(self.config.teams.red_detectors):
            idx = self.config.teams.red_fighters + i
            name = f"red_detector_{i}"
            self.possible_agents.append(name)
            self._agent_specs[name] = ("red", idx, "detector")

        for i in range(self.config.teams.blue_fighters):
            name = f"blue_fighter_{i}"
            self.possible_agents.append(name)
            self._agent_specs[name] = ("blue", i, "fighter")
        for i in range(self.config.teams.blue_detectors):
            idx = self.config.teams.blue_fighters + i
            name = f"blue_detector_{i}"
            self.possible_agents.append(name)
            self._agent_specs[name] = ("blue", idx, "detector")

    def _obs_dim_for_side(self, side: str) -> int:
        if side == "red":
            own_n = self.config.teams.red_fighters + self.config.teams.red_detectors
            enemy_n = self.config.teams.blue_fighters + self.config.teams.blue_detectors
        else:
            own_n = self.config.teams.blue_fighters + self.config.teams.blue_detectors
            enemy_n = self.config.teams.red_fighters + self.config.teams.red_detectors

        self_dim = 14
        ally_dim = max(0, own_n - 1) * 10
        enemy_dim = enemy_n * 13
        mask_dim = max(0, own_n - 1) + enemy_n + 1
        global_dim = 12
        return self_dim + ally_dim + enemy_dim + mask_dim + global_dim

    def _build_spaces(self) -> None:
        red_enemy_n = self.config.teams.blue_fighters + self.config.teams.blue_detectors
        blue_enemy_n = self.config.teams.red_fighters + self.config.teams.red_detectors
        red_obs_dim = self._obs_dim_for_side("red")
        blue_obs_dim = self._obs_dim_for_side("blue")

        for agent in self.possible_agents:
            side, _, unit_kind = self._agent_specs[agent]
            obs_dim = red_obs_dim if side == "red" else blue_obs_dim
            self._obs_spaces[agent] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )

            if unit_kind == "fighter":
                max_enemy = red_enemy_n if side == "red" else blue_enemy_n
                self._act_spaces[agent] = spaces.MultiDiscrete(
                    [36, self.config.radar.freq_count + 1, self.config.radar.freq_count + 2, 3, max_enemy + 1]
                )
            else:
                self._act_spaces[agent] = spaces.MultiDiscrete([36, self.config.radar.freq_count + 1])

    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._act_spaces[agent]

    def _convert_actions(self, actions: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
        red_f = np.zeros((self.config.teams.red_fighters, 4), dtype=np.float32)
        red_d = np.zeros((self.config.teams.red_detectors, 2), dtype=np.float32)
        blue_f = np.zeros((self.config.teams.blue_fighters, 4), dtype=np.float32)
        blue_d = np.zeros((self.config.teams.blue_detectors, 2), dtype=np.float32)

        for agent, act in actions.items():
            if agent not in self._agent_specs:
                continue
            side, idx, unit_kind = self._agent_specs[agent]
            arr = np.asarray(act, dtype=np.int32)

            if unit_kind == "fighter":
                course_bin = int(arr[0]) if arr.size > 0 else 0
                radar_freq = int(arr[1]) if arr.size > 1 else 0
                jammer_freq = int(arr[2]) if arr.size > 2 else 0
                fire_type = int(arr[3]) if arr.size > 3 else 0
                target_idx = int(arr[4]) if arr.size > 4 else 0
                course = float((course_bin % 36) * 10)

                max_enemy = (
                    self.config.teams.blue_fighters + self.config.teams.blue_detectors
                    if side == "red"
                    else self.config.teams.red_fighters + self.config.teams.red_detectors
                )
                target_idx = int(np.clip(target_idx, 0, max_enemy))
                fire_code = 0
                if fire_type == 1 and target_idx > 0:
                    fire_code = target_idx
                elif fire_type == 2 and target_idx > 0:
                    fire_code = max_enemy + target_idx

                row = np.array([course, radar_freq, jammer_freq, fire_code], dtype=np.float32)
                if side == "red" and idx < len(red_f):
                    red_f[idx] = row
                if side == "blue" and idx < len(blue_f):
                    blue_f[idx] = row
            else:
                local_idx = idx - (self.config.teams.red_fighters if side == "red" else self.config.teams.blue_fighters)
                course_bin = int(arr[0]) if arr.size > 0 else 0
                radar_freq = int(arr[1]) if arr.size > 1 else 0
                row = np.array([float((course_bin % 36) * 10), radar_freq], dtype=np.float32)
                if side == "red" and 0 <= local_idx < len(red_d):
                    red_d[local_idx] = row
                if side == "blue" and 0 <= local_idx < len(blue_d):
                    blue_d[local_idx] = row

        return {
            "red": {"fighter_action": red_f, "detector_action": red_d},
            "blue": {"fighter_action": blue_f, "detector_action": blue_d},
        }

    def _build_agent_obs(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for agent in self.agents:
            side, idx, _ = self._agent_specs[agent]
            side_obs = obs[side]["modern"]
            out[agent] = _flatten_agent_obs(side_obs, idx)
        return out

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        obs, info = self.engine.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        agent_obs = self._build_agent_obs(obs)
        infos = {agent: {"winner": info.get("winner", "ongoing")} for agent in self.agents}
        return agent_obs, infos

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        maca_actions = self._convert_actions(actions)
        obs, reward, done, truncated, info = self.engine.step(maca_actions)

        red_units = reward["red_unit"]
        blue_units = reward["blue_unit"]

        obs_out: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terminated: dict[str, bool] = {}
        trunc: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}

        current_agents = self.agents[:]
        for agent in current_agents:
            side, idx, _ = self._agent_specs[agent]
            side_obs = obs[side]["modern"]
            obs_out[agent] = _flatten_agent_obs(side_obs, idx)
            rewards[agent] = float(red_units[idx] if side == "red" else blue_units[idx])
            terminated[agent] = bool(done and not truncated)
            trunc[agent] = bool(done and truncated)
            infos[agent] = {
                "winner": info.get("winner", "ongoing"),
                "reason": info.get("reason", ""),
                "metrics": info.get("metrics", {}),
            }

        if done:
            self.agents = []

        return obs_out, rewards, terminated, trunc, infos

    def render(self) -> np.ndarray | None:
        return self.engine.render(self.render_mode)

    def close(self) -> None:
        self.engine.close()
