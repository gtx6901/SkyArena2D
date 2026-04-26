"""SkyArena MAPPO environment adapter.

Wraps SkyArenaEngine for MAPPO training.

Status: experimental but runnable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from skyarena2d.adapters.action_types import SkyArenaSideAction
from skyarena2d.core.config import load_config
from skyarena2d.core.engine import SkyArenaEngine
from skyarena2d.opponents import RULES
from skyarena2d.training.action_adapter import SkyArenaActionAdapter
from skyarena2d.training.obs_builder import SkyArenaTrainingObsBuilder


class SkyArenaMAPPOEnv:
    """MAPPO environment adapter for SkyArena2D.

    Red side is controlled by MAPPO policy.
    Blue side is controlled by rule-based opponent.

    This adapter is part of the experimental RL stack.
    """

    def __init__(self, cfg: dict, seed_offset: int = 0, deterministic_reset: bool = False):
        self.cfg = cfg
        self.seed_offset = seed_offset
        self.deterministic_reset = bool(deterministic_reset)
        self._reset_counter = 0
        self._base_seed = int(cfg["train"].get("seed", 0))
        env_cfg = cfg["env"]

        # Load engine config
        config_path = env_cfg["config_path"]
        self.engine_config = load_config(config_path)
        self.engine = SkyArenaEngine(self.engine_config)

        # Create blue opponent
        blue_rule_name = env_cfg.get("blue_rule", "fix_rule_v2")
        if blue_rule_name not in RULES:
            raise ValueError(f"Unknown blue rule: {blue_rule_name}")
        self.blue_opponent = RULES[blue_rule_name](seed=self._base_seed + seed_offset + 1000)

        # Create obs builder
        self.obs_builder = SkyArenaTrainingObsBuilder(
            num_fighters=self.engine_config.teams.red_fighters,
            candidate_slots=env_cfg.get("candidate_slots", 6),
            search_goal_grid_size=env_cfg.get("search_goal_grid_size", 8),
            map_width=self.engine_config.map.width,
            map_height=self.engine_config.map.height,
            semantic_map_size=cfg["model"].get("semantic_map_size", 100),
            track_memory_steps=env_cfg.get("track_memory_steps", 10),
        )

        # Create action adapter
        self.action_adapter = SkyArenaActionAdapter(
            candidate_slots=env_cfg.get("candidate_slots", 6),
            course_bins=16,
            search_goal_grid_size=env_cfg.get("search_goal_grid_size", 8),
            map_width=self.engine_config.map.width,
            map_height=self.engine_config.map.height,
            radar_freq=1,
            jammer_freq=env_cfg.get("default_jammer_freq", 1),
            use_jammer_strategy=env_cfg.get("use_jammer_strategy", True),
            jammer_range=env_cfg.get("jammer_range", self.engine_config.jamming.range),
            jammer_memory_steps=env_cfg.get("jammer_memory_steps", 10),
            max_jammers_per_side=env_cfg.get("max_jammers_per_side", 3),
        )

        self.red_fighter_num = self.engine_config.teams.red_fighters
        self.blue_fighter_num = self.engine_config.teams.blue_fighters
        self._last_obs = None
        self._last_info = None

    def reset(self) -> dict:
        """Reset environment and return initial policy obs for red."""
        if self.deterministic_reset:
            env_seed = self._base_seed + self.seed_offset
            opp_seed = self._base_seed + self.seed_offset + 1000
        else:
            env_seed = self._base_seed + self.seed_offset * 100000 + self._reset_counter
            opp_seed = self._base_seed + self.seed_offset * 100000 + self._reset_counter + 1000

        obs, info = self.engine.reset(seed=env_seed)
        self.blue_opponent.reset(seed=opp_seed)
        self._reset_counter += 1
        self.obs_builder.reset()
        self.action_adapter.reset_ew_state()

        self._last_obs = obs
        self._last_info = info

        # Build policy obs for red
        return self._build_policy_obs(obs, info)

    def step(self, sky_action: SkyArenaSideAction) -> tuple[dict, float, bool, dict]:
        """Step environment with red action.

        Args:
            sky_action: SkyArenaSideAction for red side

        Returns:
            (obs, reward, done, info) tuple
        """
        # Get blue action from opponent
        blue_action_dict = self.blue_opponent.act(
            self._last_obs["blue"],
            side="blue",
            step_count=self.engine.state.step_count if self.engine.state else 0,
        )

        # Convert SkyArenaSideAction to MaCA format for engine
        red_fighter_action = sky_action.to_maca_fighter_action(self.blue_fighter_num)
        red_detector_action = np.zeros((self.engine_config.teams.red_detectors, 2), dtype=np.float32)

        # Step engine: returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = self.engine.step({
            "red": {
                "fighter_action": red_fighter_action,
                "detector_action": red_detector_action,
            },
            "blue": blue_action_dict,
        })

        self._last_obs = obs
        self._last_info = info

        # Compute red team reward
        red_reward = float(reward.get("red", 0.0))
        done = terminated or truncated

        # Build next policy obs
        policy_obs = self._build_policy_obs(obs, info)

        return policy_obs, red_reward, done, info

    def _build_policy_obs(self, obs: dict, info: dict) -> dict:
        """Build policy observation for red side."""
        # Use cached visible/fireable matrices from engine state
        cache = self.engine.state.cache
        if cache is not None:
            visible_matrix = cache.red_visible[:self.red_fighter_num, :self.blue_fighter_num]
            fireable_long = cache.red_fireable_long[:self.red_fighter_num, :self.blue_fighter_num]
            fireable_short = cache.red_fireable_short[:self.red_fighter_num, :self.blue_fighter_num]
        else:
            visible_matrix = np.zeros((self.red_fighter_num, self.blue_fighter_num), dtype=bool)
            fireable_long = np.zeros((self.red_fighter_num, self.blue_fighter_num), dtype=bool)
            fireable_short = np.zeros((self.red_fighter_num, self.blue_fighter_num), dtype=bool)

        # Build policy obs
        policy_obs = self.obs_builder.build_policy_obs(
            own=self.engine.state.red,
            enemy=self.engine.state.blue,
            visible_matrix=visible_matrix,
            fireable_long=fireable_long,
            fireable_short=fireable_short,
            step_count=self.engine.state.step_count,
            max_steps=self.engine_config.max_steps,
            current_search_goal_id=np.zeros(self.red_fighter_num, dtype=np.int64),
        )

        # Add global state for critic
        policy_obs["global_state"] = self.obs_builder.build_global_state(
            red=self.engine.state.red,
            blue=self.engine.state.blue,
            step_count=self.engine.state.step_count,
            max_steps=self.engine_config.max_steps,
        )

        return policy_obs

    def obs_shapes(self) -> dict:
        """Return observation shapes for building actor/critic."""
        return {
            "self_features": (self.red_fighter_num, 20),
            "entity_features": (self.red_fighter_num, self.obs_builder.candidate_slots, 10),
            "semantic_map": (self.red_fighter_num, 9, self.obs_builder.semantic_map_size, self.obs_builder.semantic_map_size),
            "region_features": (self.red_fighter_num, self.obs_builder.search_goal_grid_size ** 2, 10),
            "global_state": (181,),
        }

    @property
    def global_state_dim(self) -> int:
        return 181
