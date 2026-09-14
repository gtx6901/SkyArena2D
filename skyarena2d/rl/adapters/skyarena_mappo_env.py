"""SkyArena MAPPO environment adapter.

Wraps SkyArenaEngine with the Baseline V2 training contract.
"""
from __future__ import annotations

import numpy as np

from skyarena2d.adapters.action_types import SkyArenaSideAction
from skyarena2d.core.config import load_config
from skyarena2d.core.engine import SkyArenaEngine
from skyarena2d.opponents import RULES, RuleOpponentPool
from skyarena2d.training.obs_builder import SkyArenaTrainingObsBuilder


class SkyArenaMAPPOEnv:
    """MAPPO environment adapter for SkyArena2D.

    Red side is controlled by MAPPO policy.
    Blue side is controlled by rule-based opponent.

    The red policy uses compact entity observations; blue is rule-controlled.
    """

    def __init__(
        self,
        cfg: dict,
        seed_offset: int = 0,
        deterministic_reset: bool = False,
        opponent_pool: RuleOpponentPool | None = None,
    ):
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

        # Create the blue opponent. Training environments may share a pool so
        # outcomes collected by every worker affect subsequent sampling.
        blue_rule_name = env_cfg.get("blue_rule", "fix_rule_v2")
        if blue_rule_name not in RULES:
            raise ValueError(f"Unknown blue rule: {blue_rule_name}")
        # Pass map dimensions for rules that need boundary awareness
        opponent_kwargs: dict = {"seed": self._base_seed + seed_offset + 1000}
        if blue_rule_name == "no_attack_rule":
            opponent_kwargs["map_width"] = self.engine_config.map.width
            opponent_kwargs["map_height"] = self.engine_config.map.height
        self.blue_rule_name = blue_rule_name
        pool_names = list(env_cfg.get("opponent_pool", []))
        self.opponent_pool = opponent_pool
        if self.opponent_pool is None and pool_names and not self.deterministic_reset:
            kwargs_by_name = {
                "no_attack_rule": {
                    "map_width": self.engine_config.map.width,
                    "map_height": self.engine_config.map.height,
                }
            }
            self.opponent_pool = RuleOpponentPool(
                pool_names,
                RULES,
                seed=self._base_seed + seed_offset + 7000,
                uniform_mix=float(env_cfg.get("opponent_pool_uniform_mix", 0.15)),
                kwargs_by_name=kwargs_by_name,
            )
        self.current_opponent_name = blue_rule_name
        self.blue_opponent = RULES[blue_rule_name](**opponent_kwargs)

        # Create obs builder
        self.obs_builder = SkyArenaTrainingObsBuilder(
            num_fighters=self.engine_config.teams.red_fighters,
            candidate_slots=env_cfg.get("candidate_slots", 6),
            map_width=self.engine_config.map.width,
            map_height=self.engine_config.map.height,
            track_memory_steps=env_cfg.get("track_memory_steps", 10),
        )

        self.red_fighter_num = self.engine_config.teams.red_fighters
        self.blue_fighter_num = self.engine_config.teams.blue_fighters
        self._last_obs = None
        self._last_info = None

    def reset(self) -> dict:
        """Reset environment and return initial policy obs for red."""
        if self.deterministic_reset:
            env_seed = self._base_seed + self.seed_offset + self._reset_counter
            opp_seed = self._base_seed + self.seed_offset + self._reset_counter + 1000
        else:
            env_seed = self._base_seed + self.seed_offset * 100000 + self._reset_counter
            opp_seed = self._base_seed + self.seed_offset * 100000 + self._reset_counter + 1000

        obs, info = self.engine.reset(seed=env_seed)
        if self.opponent_pool is not None:
            self.current_opponent_name, self.blue_opponent = self.opponent_pool.sample(seed=opp_seed)
        else:
            self.current_opponent_name = self.blue_rule_name
            self.blue_opponent.reset(seed=opp_seed)
        self._reset_counter += 1
        self.obs_builder.reset()

        self._last_obs = obs
        self._last_info = info

        # Build policy obs for red
        return self._build_policy_obs()

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
        red_agent_reward = np.asarray(reward.get("red_unit", []), dtype=np.float32)[
            : self.red_fighter_num
        ].copy()
        done = terminated or truncated
        info["red_agent_reward"] = red_agent_reward
        info["opponent_name"] = self.current_opponent_name
        if done and self.opponent_pool is not None:
            self.opponent_pool.record_result(self.current_opponent_name, str(info.get("winner", "draw")))

        # Build next policy obs
        policy_obs = self._build_policy_obs()

        return policy_obs, red_reward, done, info

    def _build_policy_obs(self) -> dict:
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
            "self_features": (self.red_fighter_num, 22),
            "entity_features": (
                self.red_fighter_num,
                self.obs_builder.entity_slots,
                20,
            ),
            "global_state": (self.global_state_dim,),
        }

    @property
    def global_state_dim(self) -> int:
        return 181
