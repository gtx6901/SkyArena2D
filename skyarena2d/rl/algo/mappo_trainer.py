"""SkyArena MAPPO trainer.

Ported from MaCA-master/algo/mappo_trainer.py.

Status: experimental but runnable for smoke/integration checks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from skyarena2d.adapters.action_types import SkyArenaSideAction
from skyarena2d.training.action_adapter import SkyArenaActionAdapter

from ..adapters.skyarena_mappo_env import SkyArenaMAPPOEnv
from ..models.actor import SkyArenaActor
from ..models.critic import SkyArenaCritic
from ..utils.checkpoint import (
    ensure_run_dirs,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_run_config,
)
from ..utils.tb import build_writer, log_scalars
from .rollout import (
    RolloutBatch,
    allocate_batched_obs,
    allocate_rollout_obs,
    build_fire_mask_from_selected_targets,
    compute_gae,
    fill_batched_obs,
    masked_categorical,
    sample_policy_actions,
    stack_env_obs,
    to_torch_batch,
)
from .search_goal_manager import TeamSearchPlanner, TeamSearchPlannerConfig


class SkyArenaMAPPOTrainer:
    """Minimal recurrent MAPPO trainer for SkyArena2D.

    This trainer is intentionally kept close to current behavior and is treated
    as experimental for convergence-sensitive workloads.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.train_cfg = cfg["train"]
        self.env_cfg = cfg["env"]
        self.model_cfg = cfg["model"]
        self.logging_cfg = cfg.get("logging", {})
        self.eval_cfg = cfg.get("evaluation", {})

        self.num_envs = int(self.train_cfg.get("num_envs", 2))
        self.rollout_steps = int(self.train_cfg.get("rollout_steps", 64))
        self.total_env_steps = int(self.train_cfg.get("total_env_steps", 100000))
        self.save_interval = int(self.train_cfg.get("save_interval", 10000))
        self.eval_interval = int(self.train_cfg.get("eval_interval", 10000))
        self.policy_eval_interval = int(self.eval_cfg.get("policy_eval_interval", self.eval_interval))
        self.policy_eval_episodes = int(self.eval_cfg.get("policy_eval_episodes", 3))
        self.gui_eval_enabled = bool(self.eval_cfg.get("gui_eval_enabled", True))
        self.gui_eval_interval = int(self.eval_cfg.get("gui_eval_interval", 0))
        self.gui_eval_episodes = int(self.eval_cfg.get("gui_eval_episodes", 1))
        self.gui_eval_max_steps = int(self.eval_cfg.get("gui_eval_max_steps", 2000))
        self.gui_eval_render_mode = str(self.eval_cfg.get("gui_eval_render_mode", "rgb_array"))
        self.gui_eval_save_frames = bool(
            self.eval_cfg.get(
                "gui_eval_save_frames",
                self.eval_cfg.get("gui_eval_save_video", False),
            )
        )
        self.gui_eval_render_every = max(1, int(self.eval_cfg.get("gui_eval_render_every", 20)))
        self.gui_eval_dir = str(self.eval_cfg.get("gui_eval_dir", "gui_eval"))
        self.gui_eval_deterministic = bool(self.eval_cfg.get("gui_eval_deterministic", True))
        self.gui_eval_human = bool(self.eval_cfg.get("gui_eval_human", False))
        self.ppo_epochs = int(self.train_cfg.get("ppo_epochs", 3))
        self.gamma = float(self.train_cfg.get("gamma", 0.99))
        self.gae_lambda = float(self.train_cfg.get("gae_lambda", 0.95))
        self.clip_coef = float(self.train_cfg.get("clip_coef", 0.2))
        self.ent_coef = float(self.train_cfg.get("entropy_coef", 0.01))
        self.vf_coef = float(self.train_cfg.get("value_coef", 0.5))
        self.max_grad_norm = float(self.train_cfg.get("max_grad_norm", 0.5))
        self.search_goal_grid_size = int(self.env_cfg.get("search_goal_grid_size", 8))
        self.search_goal_bins = self.search_goal_grid_size ** 2
        self.candidate_slots = int(self.env_cfg.get("candidate_slots", 6))

        device_str = str(self.train_cfg.get("device", "cpu"))
        self.device = torch.device(device_str)

        # Build environments
        self.envs = [SkyArenaMAPPOEnv(cfg, seed_offset=i) for i in range(self.num_envs)]
        self.num_agents = self.envs[0].red_fighter_num
        obs_shapes = self.envs[0].obs_shapes()

        # Build search goal manager
        engine_cfg = self.envs[0].engine_config
        self.search_goal_manager = TeamSearchPlanner(
            TeamSearchPlannerConfig(
                num_envs=self.num_envs,
                num_agents=self.num_agents,
                map_size_x=engine_cfg.map.width,
                map_size_y=engine_cfg.map.height,
                search_goal_grid_size=self.search_goal_grid_size,
                goal_hold_steps=int(self.env_cfg.get("goal_hold_steps", 10)),
                goal_reach_radius=float(self.env_cfg.get("goal_reach_radius", 90.0)),
            )
        )

        # Build actor and critic
        self.actor = SkyArenaActor(
            self_dim=obs_shapes["self_features"][1],
            entity_dim=obs_shapes["entity_features"][2],
            map_channels=obs_shapes["semantic_map"][1],
            candidate_slots=self.candidate_slots,
            num_agents=self.num_agents,
            course_bins=16,
            search_goal_bins=self.search_goal_bins,
            region_feature_dim=obs_shapes["region_features"][2],
            trunk_dim=int(self.model_cfg.get("trunk_dim", 192)),
            lstm_hidden_dim=int(self.model_cfg.get("lstm_hidden_dim", 192)),
            entity_embed_dim=int(self.model_cfg.get("entity_embed_dim", 96)),
            map_embed_dim=int(self.model_cfg.get("map_embed_dim", 96)),
            semantic_map_size=int(self.model_cfg.get("semantic_map_size", 100)),
            current_goal_embed_dim=int(self.model_cfg.get("current_goal_embed_dim", 32)),
            agent_id_embed_dim=int(self.model_cfg.get("agent_id_embed_dim", 16)),
            region_embed_dim=int(self.model_cfg.get("region_embed_dim", 64)),
        ).to(self.device)

        self.critic = SkyArenaCritic(
            global_state_dim=obs_shapes["global_state"][0],
            hidden_dim=int(self.model_cfg.get("critic_hidden_dim", 256)),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=float(self.train_cfg.get("learning_rate", 3e-4)),
        )

        # Build action adapter
        self.action_adapter = SkyArenaActionAdapter(
            candidate_slots=self.candidate_slots,
            course_bins=16,
            search_goal_grid_size=self.search_goal_grid_size,
            map_width=engine_cfg.map.width,
            map_height=engine_cfg.map.height,
            radar_freq=self.env_cfg.get("default_radar_freq", 1),
            radar_freq_count=self.env_cfg.get("radar_freq_count", engine_cfg.radar.freq_count),
            radar_cycle_interval=self.env_cfg.get("radar_cycle_interval", 8),
            radar_stride=self.env_cfg.get("radar_stride", 3),
            jammer_freq=self.env_cfg.get("default_jammer_freq", 1),
            jammer_cycle_interval=self.env_cfg.get("jammer_cycle_interval", 6),
            jammer_stride=self.env_cfg.get("jammer_stride", 7),
            jammer_barrage_prob=self.env_cfg.get("jammer_barrage_prob", 0.0),
            use_jammer_strategy=self.env_cfg.get("use_jammer_strategy", True),
            jammer_range=self.env_cfg.get("jammer_range", engine_cfg.jamming.range),
            jammer_memory_steps=self.env_cfg.get("jammer_memory_steps", 10),
            max_jammers_per_side=self.env_cfg.get("max_jammers_per_side", 3),
        )

        # Setup run dirs
        run_train_cfg = dict(self.train_cfg)
        run_train_cfg["logging"] = self.logging_cfg
        self.run_dirs = ensure_run_dirs(run_train_cfg)
        save_run_config(self.run_dirs, cfg)

        self.env_steps = 0
        self.update_idx = 0

        init_checkpoint = str(self.train_cfg.get("init_checkpoint", "")).strip()

        # Resume from checkpoint if requested
        if bool(self.train_cfg.get("resume", False)):
            if init_checkpoint:
                print(
                    f"[init_checkpoint] warning: resume=true, ignoring init_checkpoint {init_checkpoint}",
                    flush=True,
                )
            ckpt_path = latest_checkpoint(run_train_cfg)
            if ckpt_path is not None:
                ckpt = load_checkpoint(ckpt_path, self.actor, self.critic, self.optimizer, map_location=self.device)
                self.env_steps = int(ckpt.get("env_steps", 0))
                self.update_idx = int(ckpt.get("update_idx", 0))
                print(f"[resume] loaded checkpoint {ckpt_path} env_steps={self.env_steps}", flush=True)
        elif init_checkpoint:
            load_checkpoint(init_checkpoint, self.actor, self.critic, self.optimizer, map_location=self.device)
            print(
                f"[init_checkpoint] loaded warm-start checkpoint {init_checkpoint}; env_steps reset to 0",
                flush=True,
            )

        # Initialize obs and hidden states
        self.current_obs = [env.reset() for env in self.envs]
        self.search_goal_manager.reset_all()
        self._episode_returns = np.zeros((self.num_envs,), dtype=np.float32)
        self._episode_lengths = np.zeros((self.num_envs,), dtype=np.int32)
        hidden_dim = self.actor.lstm_hidden_dim
        self.actor_h = torch.zeros((self.num_envs, self.num_agents, hidden_dim), dtype=torch.float32, device=self.device)
        self.actor_c = torch.zeros((self.num_envs, self.num_agents, hidden_dim), dtype=torch.float32, device=self.device)

        self.writer = build_writer(run_train_cfg, purge_step=(self.env_steps if self.env_steps > 0 else None))
        self.next_save_step = self.env_steps + self.save_interval
        self.next_eval_step = self.env_steps + self.policy_eval_interval
        self.next_gui_eval_step = (
            self.env_steps + self.gui_eval_interval
            if self.gui_eval_enabled and self.gui_eval_interval > 0
            else None
        )
        if self.next_gui_eval_step is not None:
            render_mode = "human" if self.gui_eval_human else self.gui_eval_render_mode
            if render_mode == "rgb_array":
                print("[gui_eval] mode=rgb_array, no window will be opened", flush=True)
                print("[gui_eval] use --gui_eval_human to open a live window", flush=True)
            elif render_mode == "human":
                print("[gui_eval] mode=human, a live window will be opened during GUI eval", flush=True)

    @staticmethod
    def _mean_numeric_dicts(dicts: List[dict]) -> dict:
        """Average numeric values across a list of dicts. Non-numeric keys are skipped."""
        if not dicts:
            return {}
        result = {}
        for key in dicts[0]:
            values = []
            for d in dicts:
                v = d.get(key)
                if v is not None and isinstance(v, (int, float, np.integer, np.floating, bool)):
                    values.append(float(v))
            if values:
                result[key] = float(np.mean(values))
        return result

    def _collect_rollout_step_diagnostics(self, batch: RolloutBatch) -> dict:
        """Compute per-step rollout diagnostics from actions / masks / obs."""
        T, E = batch.reward.shape
        N = self.num_agents

        alive_mask = (batch.observations["alive_mask"][:, :, :N] > 0.5).astype(np.float32)
        has_contact = (batch.observations["has_active_contact"][:, :, :N] > 0.5).astype(np.float32)
        entity_mask = batch.observations["entity_mask"].reshape(T, E, N, -1).astype(np.float32)
        candidate_ids = batch.observations["candidate_ids"].reshape(T, E, N, -1)
        can_long = batch.observations["candidate_can_long"].reshape(T, E, N, -1).astype(np.float32)
        can_short = batch.observations["candidate_can_short"].reshape(T, E, N, -1).astype(np.float32)
        target_action = batch.target_action[:, :, :N]
        fire_action = batch.fire_action[:, :, :N]

        alive_count = alive_mask.sum(axis=-1)  # (T, E)
        alive_rate = alive_count / max(N, 1)

        # Per-agent entity count
        entity_per_agent = entity_mask.sum(axis=-1)  # (T, E, N)
        entity_valid_count = np.where(alive_mask > 0, entity_per_agent, 0.0).sum(axis=-1) / np.maximum(alive_count, 1)

        # Contact rate among alive agents
        contact_rate = (has_contact * alive_mask).sum(axis=-1) / np.maximum(alive_count, 1)

        # Target / fire nonzero (binary per step: any alive agent selected target/fire > 0)
        target_nonzero = np.any((target_action > 0) & (alive_mask > 0.5), axis=-1)
        fire_nonzero = np.any((fire_action > 0) & (alive_mask > 0.5), axis=-1)

        target_nonzero_rate = float(target_nonzero.mean())
        fire_nonzero_rate = float(fire_nonzero.mean())

        # no_target_rate: steps with alive agents but no target
        has_alive = alive_count > 0
        no_target_rate = float(np.where(has_alive, ~target_nonzero, 0.0).mean()) if has_alive.any() else 0.0

        # no_fire_rate: steps with target but no fire
        target_steps = max(int(target_nonzero.sum()), 1)
        no_fire_rate = float((target_nonzero & ~fire_nonzero).sum()) / target_steps

        # Target availability per alive agent
        has_valid = np.any(candidate_ids >= 0, axis=-1)
        target_available = np.where(alive_mask > 0, has_valid, 0.0).sum(axis=-1) / np.maximum(alive_count, 1)

        # Long / short available rates among valid candidates
        valid = candidate_ids >= 0
        n_valid = valid.sum()
        long_count = (can_long * valid).sum()
        short_count = (can_short * valid).sum()

        return {
            "target_nonzero_rate": target_nonzero_rate,
            "fire_nonzero_rate": fire_nonzero_rate,
            "no_target_rate": no_target_rate,
            "no_fire_rate": no_fire_rate,
            "alive_rate": float(alive_rate.mean()),
            "contact_rate": float(contact_rate.mean()),
            "entity_valid_count": float(entity_valid_count.mean()),
            "target_available_rate": float(target_available.mean()),
            "long_available_rate": float(long_count / max(n_valid, 1)),
            "short_available_rate": float(short_count / max(n_valid, 1)),
        }

    def _summarize_rollout_episodes(self, episode_stats: List[Dict[str, Any]]) -> Dict[str, float]:
        """Summarize naturally finished episodes seen inside the latest rollout.

        Returns empty dict when no episodes finished (so nothing is logged).
        """
        episodes_finished = len(episode_stats)
        if episodes_finished == 0:
            return {}
        red_wins = sum(1 for s in episode_stats if s.get("winner") == "red")
        blue_wins = sum(1 for s in episode_stats if s.get("winner") == "blue")
        draws = sum(1 for s in episode_stats if s.get("winner") not in ("red", "blue"))
        avg_len = float(np.mean([float(s.get("episode_len", s.get("steps", 0))) for s in episode_stats]))
        avg_return = float(np.mean([float(s.get("episode_return", 0.0)) for s in episode_stats]))
        return {
            "win_rate": red_wins / max(episodes_finished, 1),
            "episode_len": avg_len,
            "episode_return": avg_return,
            "red_wins": float(red_wins),
            "blue_wins": float(blue_wins),
            "draws": float(draws),
        }

    @staticmethod
    def _summarize_episode_metrics(episode_stats: List[Dict[str, Any]]) -> dict:
        """Extract episode-level metrics from final info['metrics'] across episodes."""
        if not episode_stats:
            return {}
        result = {}
        for key in [
            "red_kills", "blue_kills", "red_losses", "blue_losses",
            "missiles_launched_long", "missiles_launched_short",
            "missiles_hit", "missiles_missed",
            "red_attempted_edges", "red_selected_edges", "red_invalid_fire_count",
            "contact_to_fire_gap",
        ]:
            values = [float(v) for s in episode_stats
                      if s.get("final_metrics") and (v := s["final_metrics"].get(key)) is not None]
            if values:
                result[key] = float(np.mean(values))
        return result

    def _collect_rollout(self) -> RolloutBatch:
        """Collect rollout_steps of experience from all envs."""
        example_obs = self.current_obs[0]
        rollout_obs = allocate_rollout_obs(example_obs, self.rollout_steps, self.num_envs)
        rollout_course = np.zeros((self.rollout_steps, self.num_envs, self.num_agents), dtype=np.int64)
        rollout_search_goal = np.zeros((self.rollout_steps, self.num_envs, self.num_agents), dtype=np.int64)
        rollout_search_goal_refresh = np.zeros((self.rollout_steps, self.num_envs, self.num_agents), dtype=np.bool_)
        rollout_target = np.zeros((self.rollout_steps, self.num_envs, self.num_agents), dtype=np.int64)
        rollout_fire = np.zeros((self.rollout_steps, self.num_envs, self.num_agents), dtype=np.int64)
        rollout_log_prob = np.zeros((self.rollout_steps, self.num_envs, self.num_agents), dtype=np.float32)
        rollout_reward = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        rollout_done = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        rollout_value = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        initial_h = self.actor_h.detach().cpu().numpy().copy()
        initial_c = self.actor_c.detach().cpu().numpy().copy()
        episode_stats: List[Dict[str, Any]] = []
        step_metrics_list: List[dict] = []

        for step in range(self.rollout_steps):
            obs_batch = stack_env_obs(self.current_obs)
            batched_obs = allocate_batched_obs(example_obs, self.num_envs)
            fill_batched_obs(batched_obs, self.current_obs)

            with torch.no_grad():
                sampled = sample_policy_actions(
                    self.actor,
                    batched_obs,
                    (self.actor_h, self.actor_c),
                    self.device,
                    deterministic=False,
                    search_goal_manager=self.search_goal_manager,
                )
                global_state_t = torch.as_tensor(batched_obs["global_state"], dtype=torch.float32, device=self.device)
                value = self.critic(global_state_t).detach().cpu().numpy()

            # Store obs
            for key in rollout_obs:
                rollout_obs[key][step] = batched_obs[key]
            rollout_course[step] = sampled["course"]
            rollout_search_goal[step] = sampled["search_goal"]
            rollout_search_goal_refresh[step] = sampled["search_goal_refresh_mask"]
            rollout_target[step] = sampled["target"]
            rollout_fire[step] = sampled["fire"]
            rollout_log_prob[step] = sampled["log_prob"]
            rollout_value[step] = value

            # Update hidden states
            self.actor_h = sampled["next_h"].to(self.device)
            self.actor_c = sampled["next_c"].to(self.device)

            # Step each env
            new_obs_list = []
            for env_idx, env in enumerate(self.envs):
                course_i = sampled["course"][env_idx]
                search_goal_i = sampled["search_goal"][env_idx]
                target_i = sampled["target"][env_idx]
                fire_i = sampled["fire"][env_idx]

                sky_action = self.action_adapter.decode(
                    course_action=course_i,
                    search_goal_action=search_goal_i,
                    target_action=target_i,
                    fire_action=fire_i,
                    own=env.engine.state.red,
                    candidate_ids=batched_obs["candidate_ids"][env_idx],
                    candidate_can_long=batched_obs["candidate_can_long"][env_idx],
                    candidate_can_short=batched_obs["candidate_can_short"][env_idx],
                    has_active_contact=batched_obs["has_active_contact"][env_idx] > 0.5,
                    current_heading=env.engine.state.red.heading[:env.red_fighter_num],
                    entity_features=batched_obs["entity_features"][env_idx],
                    ew_state_key=env_idx,
                    step_count=env.engine.state.step_count,
                )

                env.set_current_search_goal_id(search_goal_i)
                next_obs, reward, done, info = env.step(sky_action)
                rollout_reward[step, env_idx] = reward
                rollout_done[step, env_idx] = float(done)
                self._episode_returns[env_idx] += float(reward)
                self._episode_lengths[env_idx] += 1

                # Collect per-step metrics
                step_metrics_list.append(info.get("metrics", {}))

                if done:
                    episode_stats.append({
                        "env_idx": env_idx,
                        "winner": info.get("winner", "unknown"),
                        "reason": info.get("reason", ""),
                        "steps": env.engine.state.step_count,
                        "episode_len": int(self._episode_lengths[env_idx]),
                        "episode_return": float(self._episode_returns[env_idx]),
                        "final_metrics": info.get("metrics", {}),
                    })
                    self._episode_returns[env_idx] = 0.0
                    self._episode_lengths[env_idx] = 0
                    next_obs = env.reset()
                    self.search_goal_manager.reset_envs([env_idx])
                    self.action_adapter.reset_ew_state(env_idx)
                    self.actor_h[env_idx] = 0.0
                    self.actor_c[env_idx] = 0.0

                new_obs_list.append(next_obs)

            self.current_obs = new_obs_list
            self.env_steps += self.num_envs

        # Compute next value for GAE
        with torch.no_grad():
            final_batched = allocate_batched_obs(example_obs, self.num_envs)
            fill_batched_obs(final_batched, self.current_obs)
            global_state_t = torch.as_tensor(final_batched["global_state"], dtype=torch.float32, device=self.device)
            next_value = self.critic(global_state_t).detach().cpu().numpy()

        batch = RolloutBatch(
            observations=rollout_obs,
            initial_h=initial_h,
            initial_c=initial_c,
            course_action=rollout_course,
            search_goal_action=rollout_search_goal,
            search_goal_refresh_mask=rollout_search_goal_refresh,
            target_action=rollout_target,
            fire_action=rollout_fire,
            log_prob=rollout_log_prob,
            reward=rollout_reward,
            done=rollout_done,
            value=rollout_value,
            next_value=next_value,
            episode_stats=episode_stats,
        )
        return batch, step_metrics_list

    def _ppo_update(self, batch: RolloutBatch) -> Dict[str, float]:
        """Run PPO update on collected rollout batch."""
        advantages, returns = compute_gae(
            batch.reward, batch.value, batch.next_value, batch.done,
            self.gamma, self.gae_lambda,
        )
        # Normalize advantages
        adv_flat = advantages.flatten()
        adv_mean = adv_flat.mean()
        adv_std = adv_flat.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        T, E = batch.reward.shape
        N = self.num_agents
        total_loss_sum = 0.0
        pg_loss_sum = 0.0
        vf_loss_sum = 0.0
        ent_loss_sum = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            # Flatten T*E for batch processing
            obs_flat = {
                key: torch.as_tensor(
                    val.reshape(T * E, *val.shape[2:]), dtype=torch.float32, device=self.device
                )
                for key, val in batch.observations.items()
                if key not in ("global_state",)
            }
            obs_flat["global_state"] = torch.as_tensor(
                batch.observations["global_state"].reshape(T * E, -1), dtype=torch.float32, device=self.device
            )

            # Expand obs for agents: (T*E, N, ...) -> (T*E*N, ...)
            self_feat = obs_flat["self_features"].reshape(T * E, N, -1)
            entity_feat = obs_flat["entity_features"].reshape(T * E, N, *batch.observations["entity_features"].shape[3:])
            entity_mask = obs_flat["entity_mask"].reshape(T * E, N, -1)
            sem_map = obs_flat["semantic_map"].reshape(T * E, N, *batch.observations["semantic_map"].shape[3:])
            goal_id = torch.as_tensor(batch.observations["current_search_goal_id"].reshape(T * E, N), dtype=torch.long, device=self.device)
            agent_id = torch.as_tensor(batch.observations["agent_id"].reshape(T * E, N), dtype=torch.long, device=self.device)
            region_feat = obs_flat["region_features"].reshape(T * E, N, *batch.observations["region_features"].shape[3:])

            flat_actor_batch = {
                "self_features": self_feat.reshape(T * E * N, -1),
                "entity_features": entity_feat.reshape(T * E * N, *entity_feat.shape[2:]),
                "entity_mask": entity_mask.reshape(T * E * N, -1),
                "semantic_map": sem_map.reshape(T * E * N, *sem_map.shape[2:]),
                "current_search_goal_id": goal_id.reshape(T * E * N),
                "agent_id": agent_id.reshape(T * E * N),
                "region_features": region_feat.reshape(T * E * N, *region_feat.shape[2:]),
            }

            h0 = torch.as_tensor(batch.initial_h.reshape(E * N, -1), dtype=torch.float32, device=self.device)
            c0 = torch.as_tensor(batch.initial_c.reshape(E * N, -1), dtype=torch.float32, device=self.device)

            # Forward pass through actor (step-by-step for LSTM)
            all_course_logits = []
            all_sg_logits = []
            all_target_logits = []
            all_fire_logits = []
            h, c = h0, c0
            for t in range(T):
                step_batch = {k: v[t * E * N:(t + 1) * E * N] for k, v in flat_actor_batch.items()}
                out = self.actor.step(step_batch, (h, c))
                all_course_logits.append(out["course_logits"])
                all_sg_logits.append(out["search_goal_logits"])
                all_target_logits.append(out["target_logits"])
                all_fire_logits.append(out["fire_logits"])
                h, c = out["next_h"], out["next_c"]

            course_logits = torch.stack(all_course_logits, dim=0).reshape(T, E, N, -1)
            sg_logits = torch.stack(all_sg_logits, dim=0).reshape(T, E, N, -1)
            target_logits = torch.stack(all_target_logits, dim=0).reshape(T, E, N, -1)
            fire_logits_all = torch.stack(all_fire_logits, dim=0).reshape(T, E, N, -1, 3)

            # Compute log probs
            course_mask = torch.as_tensor(batch.observations["course_mask"].reshape(T, E, N, -1), dtype=torch.bool, device=self.device)
            sg_mask = torch.as_tensor(batch.observations["search_goal_mask"].reshape(T, E, N, -1), dtype=torch.bool, device=self.device)
            target_mask = torch.as_tensor(batch.observations["target_mask"].reshape(T, E, N, -1), dtype=torch.bool, device=self.device)
            alive_mask = torch.as_tensor(batch.observations["alive_mask"].reshape(T, E, N), dtype=torch.float32, device=self.device) > 0.5
            has_contact = torch.as_tensor(batch.observations["has_active_contact"].reshape(T, E, N), dtype=torch.float32, device=self.device) > 0.5
            entity_mask_t = torch.as_tensor(batch.observations["entity_mask"].reshape(T, E, N, -1), dtype=torch.bool, device=self.device)
            candidate_can_long = torch.as_tensor(batch.observations["candidate_can_long"].reshape(T, E, N, -1), dtype=torch.bool, device=self.device)
            candidate_can_short = torch.as_tensor(batch.observations["candidate_can_short"].reshape(T, E, N, -1), dtype=torch.bool, device=self.device)

            course_act = torch.as_tensor(batch.course_action.reshape(T, E, N), dtype=torch.long, device=self.device)
            sg_act = torch.as_tensor(batch.search_goal_action.reshape(T, E, N), dtype=torch.long, device=self.device)
            target_act = torch.as_tensor(batch.target_action.reshape(T, E, N), dtype=torch.long, device=self.device)
            fire_act = torch.as_tensor(batch.fire_action.reshape(T, E, N), dtype=torch.long, device=self.device)

            course_dist = masked_categorical(course_logits.reshape(T * E * N, -1), course_mask.reshape(T * E * N, -1))
            sg_dist = masked_categorical(sg_logits.reshape(T * E * N, -1), sg_mask.reshape(T * E * N, -1))
            target_dist = masked_categorical(target_logits.reshape(T * E * N, -1), target_mask.reshape(T * E * N, -1))

            course_lp = course_dist.log_prob(course_act.reshape(T * E * N)).reshape(T, E, N)
            sg_lp = sg_dist.log_prob(sg_act.reshape(T * E * N)).reshape(T, E, N)
            target_lp = target_dist.log_prob(target_act.reshape(T * E * N)).reshape(T, E, N)

            # Fire log prob: select fire logits for chosen target
            target_act_clamped = torch.clamp(target_act, min=0, max=fire_logits_all.shape[3] - 1)
            fire_logits_sel = fire_logits_all.gather(
                3, target_act_clamped.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 3)
            ).squeeze(3)
            fire_mask_t = build_fire_mask_from_selected_targets(
                target_action=target_act.reshape(T * E * N),
                alive_mask=alive_mask.reshape(T * E * N),
                candidate_can_long=candidate_can_long.reshape(T * E * N, -1),
                candidate_can_short=candidate_can_short.reshape(T * E * N, -1),
            ).reshape(T, E, N, 3)
            fire_dist = masked_categorical(fire_logits_sel.reshape(T * E * N, 3), fire_mask_t.reshape(T * E * N, 3))
            fire_lp = fire_dist.log_prob(fire_act.reshape(T * E * N)).reshape(T, E, N)

            has_target_opportunity = alive_mask & has_contact & torch.any(entity_mask_t, dim=-1)
            movement_log_prob = torch.where(
                has_contact,
                course_lp,
                torch.where(alive_mask & (~has_contact), sg_lp, torch.zeros_like(course_lp)),
            )
            attack_log_prob = target_lp + fire_lp
            new_log_prob = movement_log_prob + torch.where(
                has_target_opportunity,
                attack_log_prob,
                torch.zeros_like(attack_log_prob),
            )

            old_log_prob = torch.as_tensor(batch.log_prob.reshape(T, E, N), dtype=torch.float32, device=self.device)
            adv_t = torch.as_tensor(advantages.reshape(T, E), dtype=torch.float32, device=self.device).unsqueeze(-1).expand(-1, -1, N)

            ratio = torch.exp(new_log_prob - old_log_prob)
            pg_loss1 = -adv_t * ratio
            pg_loss2 = -adv_t * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            global_state_t = torch.as_tensor(batch.observations["global_state"].reshape(T * E, -1), dtype=torch.float32, device=self.device)
            new_value = self.critic(global_state_t).reshape(T, E)
            returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
            vf_loss = F.mse_loss(new_value, returns_t)

            # Entropy
            ent = (course_dist.entropy() + sg_dist.entropy()).mean()

            loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

            total_loss_sum += loss.item()
            pg_loss_sum += pg_loss.item()
            vf_loss_sum += vf_loss.item()
            ent_loss_sum += ent.item()
            n_updates += 1

        self.update_idx += 1
        return {
            "loss": total_loss_sum / max(n_updates, 1),
            "pg_loss": pg_loss_sum / max(n_updates, 1),
            "vf_loss": vf_loss_sum / max(n_updates, 1),
            "entropy": ent_loss_sum / max(n_updates, 1),
        }

    def train(self) -> None:
        """Main training loop."""
        print(f"[train] Starting SkyArena MAPPO training. total_env_steps={self.total_env_steps}", flush=True)
        while self.env_steps < self.total_env_steps:
            batch, step_metrics_list = self._collect_rollout()
            metrics = self._ppo_update(batch)

            # --- Compute logging groups ---
            # train/* (always written)
            log_scalars(self.writer, "train", metrics, self.env_steps)

            # rollout_step/* (always written — per-step averages)
            step_diags = self._collect_rollout_step_diagnostics(batch)
            log_scalars(self.writer, "rollout_step", step_diags, self.env_steps)

            # metrics_step/* (always written — from env info["metrics"])
            metrics_step = self._mean_numeric_dicts(step_metrics_list)
            log_scalars(self.writer, "metrics_step", metrics_step, self.env_steps)

            # episode_mean/* (only when episodes finished)
            episode_metrics = self._summarize_rollout_episodes(batch.episode_stats)
            if episode_metrics:
                log_scalars(self.writer, "episode_mean", episode_metrics, self.env_steps)

            # episode/* (from final info["metrics"] of finished episodes)
            episode_final = self._summarize_episode_metrics(batch.episode_stats)
            if episode_final:
                log_scalars(self.writer, "episode", episode_final, self.env_steps)

            print(
                f"[train] steps={self.env_steps} loss={metrics['loss']:.4f} "
                f"pg={metrics['pg_loss']:.4f} vf={metrics['vf_loss']:.4f} "
                f"ent={metrics['entropy']:.4f}",
                flush=True,
            )
            if episode_metrics:
                n_finished = len(batch.episode_stats)
                print(
                    f"[rollout] finished={n_finished} "
                    f"win_rate={episode_metrics['win_rate']:.3f} "
                    f"avg_len={episode_metrics['episode_len']:.1f}",
                    flush=True,
                )
            else:
                print("[rollout] no episodes finished this update", flush=True)

            if self.env_steps >= self.next_save_step:
                path = save_checkpoint(
                    train_cfg=self.train_cfg,
                    actor=self.actor,
                    critic=self.critic,
                    optimizer=self.optimizer,
                    env_steps=self.env_steps,
                    update_idx=self.update_idx,
                )
                print(f"[checkpoint] saved {path}", flush=True)
                self.next_save_step += self.save_interval

            if self.env_steps >= self.next_eval_step:
                eval_metrics = self.evaluate(
                    num_episodes=self.policy_eval_episodes,
                    deterministic=True,
                    deterministic_reset=True,
                    write_report=True,
                    kind="eval",
                )
                log_scalars(self.writer, "eval", eval_metrics, self.env_steps)
                self.next_eval_step += self.policy_eval_interval

            if self.next_gui_eval_step is not None and self.env_steps >= self.next_gui_eval_step:
                render_mode = "human" if self.gui_eval_human else self.gui_eval_render_mode
                save_frames = bool(render_mode == "rgb_array" and self.gui_eval_save_frames)
                output_dir = None
                if save_frames:
                    output_dir = (
                        self.run_dirs["exp_dir"]
                        / self.gui_eval_dir
                        / f"step_{self.env_steps:09d}"
                    )
                gui_metrics = self.evaluate(
                    num_episodes=self.gui_eval_episodes,
                    deterministic=self.gui_eval_deterministic,
                    deterministic_reset=self.gui_eval_deterministic,
                    render_mode=render_mode,
                    max_steps=self.gui_eval_max_steps,
                    output_dir=output_dir,
                    save_visual=save_frames,
                    render_every=self.gui_eval_render_every,
                    step_tag=self.env_steps,
                    write_report=True,
                    kind="gui_eval",
                )
                log_scalars(self.writer, "gui_eval", gui_metrics, self.env_steps)
                if output_dir is not None:
                    print(f"[gui_eval] frame_artifacts={output_dir}", flush=True)
                self.next_gui_eval_step += self.gui_eval_interval

        latest_path = latest_checkpoint(self.train_cfg)
        latest_steps = -1
        if latest_path is not None:
            try:
                latest_steps = int(latest_path.stem.removeprefix("step_"))
            except ValueError:
                latest_steps = -1
        if latest_steps < self.env_steps:
            path = save_checkpoint(
                train_cfg=self.train_cfg,
                actor=self.actor,
                critic=self.critic,
                optimizer=self.optimizer,
                env_steps=self.env_steps,
                update_idx=self.update_idx,
            )
            print(f"[checkpoint] saved final {path}", flush=True)

        print(f"[train] Done. env_steps={self.env_steps}", flush=True)

    def _write_eval_report(
        self,
        kind: str,
        summary: dict,
        episodes: list,
        checkpoint_path: str | None = None,
    ) -> None:
        """Write eval/gui_eval results as JSON to <exp_dir>/eval_reports/."""
        import json
        import time

        report_dir = self.run_dirs["exp_dir"] / "eval_reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        env_steps = summary.get("env_steps")
        if env_steps is not None:
            fname = f"{kind}_step_{int(env_steps):09d}.json"
        else:
            fname = f"{kind}_standalone_{int(time.time())}.json"

        config_block = {
            "experiment_name": str(self.cfg["train"].get("experiment_name", "")),
            "blue_rule": str(self.env_cfg.get("blue_rule", "fix_rule_v2")),
            "num_episodes": len(episodes),
            "deterministic": summary.get("deterministic", None),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        }

        def _safe(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.item()
            if isinstance(obj, Path):
                return str(obj)
            return obj

        report = {
            "kind": kind,
            "env_steps": int(env_steps) if env_steps is not None else None,
            "update_idx": int(summary.get("update_idx")) if summary.get("update_idx") is not None else None,
            "timestamp_unix": time.time(),
            "config": config_block,
            "summary": summary,
            "episodes": episodes,
        }
        path = report_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=_safe)
        print(f"[eval_report] wrote {path}", flush=True)

    def evaluate(
        self,
        num_episodes: int = 10,
        checkpoint_path: Optional[str] = None,
        *,
        deterministic: bool = True,
        deterministic_reset: bool = True,
        render_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
        output_dir: Optional[Path] = None,
        save_visual: bool = False,
        render_every: int = 1,
        step_tag: Optional[int] = None,
        write_report: bool = False,
        kind: str = "eval",
    ) -> Dict[str, float]:
        """Evaluate current policy."""
        if checkpoint_path is not None:
            load_checkpoint(checkpoint_path, self.actor, self.critic, map_location=self.device)

        eval_env = SkyArenaMAPPOEnv(
            self.cfg,
            seed_offset=9999,
            deterministic_reset=deterministic_reset,
        )
        if render_mode is not None:
            eval_env.engine.render_mode = render_mode

        hidden_dim = self.actor.lstm_hidden_dim
        red_wins = 0
        blue_wins = 0
        draws = 0
        truncated_count = 0
        total_steps = 0
        total_return = 0.0
        total_red_kills = 0.0
        total_blue_kills = 0.0

        if save_visual and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        eval_prefix = "gui_eval" if render_mode is not None else "eval"
        step_value = int(self.env_steps if step_tag is None else step_tag)
        render_every = max(1, int(render_every))
        mode_text = render_mode if render_mode is not None else "policy"
        if eval_prefix == "gui_eval":
            print(
                f"[gui_eval] start step={step_value} episodes={num_episodes} mode={mode_text}",
                flush=True,
            )
            if render_mode == "rgb_array":
                print("[gui_eval] mode=rgb_array, no window will be opened", flush=True)
                print("[gui_eval] use --gui_eval_human to open a live window", flush=True)
        else:
            print(f"[eval] start step={step_value}", flush=True)

        # Diagnostic accumulators (reset per-episode, averaged over episodes)
        diag_target_nonzero_steps = 0
        diag_fire_nonzero_steps = 0
        diag_total_steps_diag = 0
        diag_attempted_edges = 0
        diag_selected_edges = 0
        diag_invalid_fire = 0
        diag_fireable_edges = 0
        diag_fireable_agents = 0
        diag_fireable_count = 0
        diag_valid_candidates = 0
        diag_valid_candidate_steps = 0
        diag_long_available = 0
        diag_short_available = 0
        diag_candidate_pairs = 0
        diag_missiles_remaining = 0
        diag_missile_record_count = 0
        diag_missiles_long = 0
        diag_missiles_short = 0
        diag_missiles_hit = 0
        diag_missiles_missed = 0
        diag_sel_exch = 0.0
        diag_blue_attempted = 0
        diag_blue_selected = 0
        diag_blue_invalid = 0
        diag_blue_fireable = 0
        diag_blue_fireable_count = 0
        diag_sel_exch_steps = 0
        ep_sel_exch_sum = 0.0
        # Fire head diagnostics accumulators
        diag_fire_prob_no_fire = 0.0
        diag_fire_prob_long = 0.0
        diag_fire_prob_short = 0.0
        diag_fire_entropy = 0.0
        diag_fire_prob_steps = 0
        diag_fire_mask_long = 0.0
        diag_fire_mask_short = 0.0
        diag_fire_argmax_nonzero = 0.0
        diag_adapter_zeroed_fire = 0
        diag_fire_nonzero_count = 0
        diag_target_nonzero_count = 0
        diag_tgt_can_long_sum = 0.0
        diag_tgt_can_short_sum = 0.0
        episode_records: list = []

        for ep in range(num_episodes):
            if eval_prefix == "gui_eval":
                print(f"[gui_eval] episode {ep + 1}/{num_episodes} start", flush=True)
            obs = eval_env.reset()
            self.action_adapter.reset_ew_state("eval")
            h = torch.zeros((1, self.num_agents, hidden_dim), dtype=torch.float32, device=self.device)
            c = torch.zeros((1, self.num_agents, hidden_dim), dtype=torch.float32, device=self.device)
            done = False
            ep_steps = 0
            ep_return = 0.0
            info: Dict[str, Any] = {}
            episode_truncated = False

            # Per-episode diagnostic accumulators
            ep_target_nonzero_steps = 0
            ep_fire_nonzero_steps = 0
            ep_attempted_edges = 0
            ep_selected_edges = 0
            ep_invalid_fire = 0
            ep_fireable_edges = 0
            ep_fireable_agents = 0
            ep_fireable_count = 0
            # Blue per-episode accumulators (from cache, like red)
            ep_blue_attempted = 0
            ep_blue_selected = 0
            ep_blue_invalid = 0
            ep_blue_fireable = 0
            ep_blue_fireable_count = 0
            ep_sel_exch_sum = 0.0
            ep_sel_exch_steps = 0
            # Fire head diagnostics per-episode
            ep_fire_prob_no_fire = 0.0
            ep_fire_prob_long = 0.0
            ep_fire_prob_short = 0.0
            ep_fire_entropy = 0.0
            ep_fire_prob_steps = 0
            ep_fire_mask_long = 0.0
            ep_fire_mask_short = 0.0
            ep_fire_argmax_nonzero = 0.0
            # Adapter zeroing diagnostics
            ep_adapter_zeroed_fire = 0
            ep_fire_nonzero_count = 0
            ep_target_nonzero_count = 0
            ep_tgt_can_long_sum = 0.0
            ep_tgt_can_short_sum = 0.0

            frame_dir: Optional[Path] = None
            if save_visual and render_mode == "rgb_array" and output_dir is not None:
                stem = f"step_{step_value:09d}_episode_{ep:03d}"
                frame_dir = output_dir / f"{stem}_frames"
                frame_dir.mkdir(parents=True, exist_ok=True)

            while not done and (max_steps is None or ep_steps < max_steps):
                obs_batch = {k: v[np.newaxis] for k, v in obs.items()}
                with torch.no_grad():
                    sampled = sample_policy_actions(
                        self.actor, obs_batch, (h, c), self.device, deterministic=deterministic,
                        return_diagnostics=True,
                    )
                sky_action = self.action_adapter.decode(
                    course_action=sampled["course"][0],
                    search_goal_action=sampled["search_goal"][0],
                    target_action=sampled["target"][0],
                    fire_action=sampled["fire"][0],
                    own=eval_env.engine.state.red,
                    candidate_ids=obs_batch["candidate_ids"][0],
                    candidate_can_long=obs_batch["candidate_can_long"][0],
                    candidate_can_short=obs_batch["candidate_can_short"][0],
                    has_active_contact=obs_batch["has_active_contact"][0] > 0.5,
                    current_heading=eval_env.engine.state.red.heading[:eval_env.red_fighter_num],
                    entity_features=obs_batch["entity_features"][0],
                    ew_state_key="eval",
                    step_count=eval_env.engine.state.step_count,
                )
                # --- Adapter zeroing diagnostics ---
                fire_policy = sampled["fire"][0]  # (N,) raw policy fire action
                fire_decoded = sky_action.fire_type  # (N,) decoded fire type
                tgt_policy = sampled["target"][0]  # (N,) 1-indexed slot
                tgt_decoded = sky_action.target_idx  # (N,) enemy index
                alive = (obs_batch["alive_mask"][0] > 0.5)
                can_long_arr = obs_batch["candidate_can_long"][0]
                can_short_arr = obs_batch["candidate_can_short"][0]
                # adapter zeroed: policy wanted fire but adapter set fire_type=0
                adapter_zeroed = (fire_policy > 0) & (fire_decoded == 0) & alive
                ep_adapter_zeroed_fire += int(np.count_nonzero(adapter_zeroed))
                ep_fire_nonzero_count += int(np.count_nonzero((fire_policy > 0) & alive))
                has_tgt = (tgt_policy > 0) & alive
                n_tgt = int(np.count_nonzero(has_tgt))
                ep_target_nonzero_count += n_tgt
                if n_tgt > 0:
                    slot_idx = np.clip(tgt_policy[has_tgt] - 1, 0, can_long_arr.shape[1] - 1)
                    idx = np.arange(has_tgt.shape[0])[has_tgt]
                    ep_tgt_can_long_sum += float(np.mean(can_long_arr[idx, slot_idx]))
                    ep_tgt_can_short_sum += float(np.mean(can_short_arr[idx, slot_idx]))
                eval_env.set_current_search_goal_id(sampled["search_goal"][0])
                obs, reward, done, info = eval_env.step(sky_action)
                ep_return += float(reward)
                h = sampled["next_h"].to(self.device)
                c = sampled["next_c"].to(self.device)
                ep_steps += 1

                # --- Diagnostics collection per step ---
                alive_mask = (obs_batch["alive_mask"][0] > 0.5)
                target = sampled["target"][0]
                fire = sampled["fire"][0]
                if np.any(alive_mask):
                    if np.any(target[alive_mask] > 0):
                        ep_target_nonzero_steps += 1
                    if np.any(fire[alive_mask] > 0):
                        ep_fire_nonzero_steps += 1

                cache = eval_env.engine.state.cache
                if cache is not None:
                    attempted = int(np.count_nonzero(
                        cache.red_attempted_long_matrix | cache.red_attempted_short_matrix
                    )) if cache.red_attempted_long_matrix.size > 0 else 0
                    selected = int(np.count_nonzero(
                        cache.red_selected_long_matrix | cache.red_selected_short_matrix
                    )) if cache.red_selected_long_matrix.size > 0 else 0
                    ep_attempted_edges += attempted
                    ep_selected_edges += selected
                    # invalid = attempted but not selected
                    ep_invalid_fire += max(0, attempted - selected)

                    fe = int(np.count_nonzero(
                        cache.red_fireable_long | cache.red_fireable_short
                    ))
                    fa = int(np.count_nonzero(
                        np.any(cache.red_fireable_long | cache.red_fireable_short, axis=1)
                    ))
                    ep_fireable_edges += fe
                    ep_fireable_agents += fa
                    ep_fireable_count += 1
                    # Blue metrics from cache (like red)
                    b_att = int(np.count_nonzero(
                        cache.blue_attempted_long_matrix | cache.blue_attempted_short_matrix
                    )) if cache.blue_attempted_long_matrix.size > 0 else 0
                    b_sel = int(np.count_nonzero(
                        cache.blue_selected_long_matrix | cache.blue_selected_short_matrix
                    )) if cache.blue_selected_long_matrix.size > 0 else 0
                    ep_blue_attempted += b_att
                    ep_blue_selected += b_sel
                    ep_blue_invalid += max(0, b_att - b_sel)
                    b_fe = int(np.count_nonzero(
                        cache.blue_fireable_long | cache.blue_fireable_short
                    ))
                    ep_blue_fireable += b_fe
                    ep_blue_fireable_count += 1
                    # selected_expected_exchange per step
                    sm = info.get("metrics", {})
                    sel_ex = sm.get("selected_expected_exchange")
                    if sel_ex is not None:
                        ep_sel_exch_sum += float(sel_ex)
                        ep_sel_exch_steps += 1

                # --- Fire probability diagnostics per step ---
                fire_logits_sel = sampled.get("fire_logits_selected")
                fire_raw_mask = sampled.get("fire_mask")
                if fire_logits_sel is not None and fire_raw_mask is not None and np.any(alive_mask):
                    logits = fire_logits_sel[0].astype(np.float64)  # (N, 3)
                    fmask = fire_raw_mask[0]  # (N, 3)
                    logits = logits - logits.max(axis=-1, keepdims=True)
                    exp_l = np.exp(logits)
                    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)  # (N, 3)
                    alive_probs = probs[alive_mask]
                    alive_fmask = fmask[alive_mask]
                    if alive_probs.shape[0] > 0:
                        ep_fire_prob_no_fire += float(alive_probs[:, 0].mean())
                        ep_fire_prob_long += float(alive_probs[:, 1].mean())
                        ep_fire_prob_short += float(alive_probs[:, 2].mean())
                        eps = 1e-12
                        ep_fire_entropy += float((-alive_probs * np.log(alive_probs + eps)).sum(axis=-1).mean())
                        ep_fire_prob_steps += 1
                        ep_fire_mask_long += float(alive_fmask[:, 1].mean())
                        ep_fire_mask_short += float(alive_fmask[:, 2].mean())
                        fire_argmax = np.argmax(logits[alive_mask], axis=-1)
                        ep_fire_argmax_nonzero += float(np.mean(fire_argmax > 0))

                # Candidate data from obs (before step)
                cid = obs_batch["candidate_ids"][0]
                can_long = obs_batch["candidate_can_long"][0]
                can_short = obs_batch["candidate_can_short"][0]
                valid_mask = cid >= 0
                n_valid = int(np.count_nonzero(valid_mask))
                diag_valid_candidates += n_valid
                diag_valid_candidate_steps += 1
                if n_valid > 0:
                    diag_long_available += int(np.count_nonzero(can_long & valid_mask))
                    diag_short_available += int(np.count_nonzero(can_short & valid_mask))
                    diag_candidate_pairs += n_valid

                should_render = False
                if render_mode == "human":
                    should_render = True
                elif render_mode is not None:
                    should_render = ep_steps == 1 or (ep_steps % render_every == 0)
                if should_render:
                    frame = eval_env.engine.render(render_mode)
                    if frame is not None and save_visual:
                        if frame_dir is not None:
                            np.savez_compressed(frame_dir / f"frame_{ep_steps:05d}.npz", frame=frame)

            if not done and max_steps is not None:
                episode_truncated = True

            if render_mode is not None and render_mode != "human":
                frame = eval_env.engine.render(render_mode)
                if frame is not None and save_visual and frame_dir is not None:
                    np.savez_compressed(frame_dir / f"frame_{ep_steps:05d}_final.npz", frame=frame)

            winner = "draw" if episode_truncated else str(info.get("winner", "draw"))
            if winner == "red":
                red_wins += 1
            elif winner == "blue":
                blue_wins += 1
            else:
                draws += 1
            if episode_truncated:
                truncated_count += 1
            total_steps += ep_steps
            total_return += ep_return
            metrics = info.get("metrics", {})
            total_red_kills += float(metrics.get("red_kills", 0.0))
            total_blue_kills += float(metrics.get("blue_kills", 0.0))
            if eval_prefix == "gui_eval":
                status = "truncated" if episode_truncated else winner
                print(
                    f"[gui_eval] episode {ep + 1}/{num_episodes} done "
                    f"steps={ep_steps} winner={status}",
                    flush=True,
                )

            # --- Per-episode diagnostics accumulation ---
            nz_steps = max(ep_steps, 1)
            ep_target_rate = ep_target_nonzero_steps / nz_steps
            ep_fire_rate = ep_fire_nonzero_steps / nz_steps
            ep_attempted_mean = ep_attempted_edges / nz_steps
            ep_selected_mean = ep_selected_edges / nz_steps
            ep_invalid_mean = ep_invalid_fire / nz_steps
            ep_fireable_mean = ep_fireable_edges / max(ep_fireable_count, 1)
            ep_red_missiles = int(np.sum(eval_env.engine.state.red.long_ammo[:eval_env.red_fighter_num])
                                  + np.sum(eval_env.engine.state.red.short_ammo[:eval_env.red_fighter_num]))
            ep_blue_missiles = int(np.sum(eval_env.engine.state.blue.long_ammo[:eval_env.blue_fighter_num])
                                   + np.sum(eval_env.engine.state.blue.short_ammo[:eval_env.blue_fighter_num]))

            # Blue step-mean rates
            ep_blue_attempted_mean = ep_blue_attempted / nz_steps
            ep_blue_selected_mean = ep_blue_selected / nz_steps
            ep_blue_fireable_mean = ep_blue_fireable / max(ep_blue_fireable_count, 1)
            ep_blue_invalid_mean = ep_blue_invalid / nz_steps
            # selected_expected_exchange step mean
            ep_sel_exch_mean = ep_sel_exch_sum / max(ep_sel_exch_steps, 1)

            # Fire prob means
            fp_denom = max(ep_fire_prob_steps, 1)
            ep_fire_noop_mean = ep_fire_prob_no_fire / fp_denom
            ep_fire_long_mean = ep_fire_prob_long / fp_denom
            ep_fire_short_mean = ep_fire_prob_short / fp_denom
            ep_fire_ent_mean = ep_fire_entropy / fp_denom
            ep_fire_mask_long_rate = ep_fire_mask_long / fp_denom
            ep_fire_mask_short_rate = ep_fire_mask_short / fp_denom
            ep_fire_argmax_nz_rate = ep_fire_argmax_nonzero / fp_denom
            # Adapter diag rates
            ep_adapter_zeroed_rate = ep_adapter_zeroed_fire / max(ep_fire_nonzero_count, 1)
            ep_tgt_fireable_steps = max(ep_target_nonzero_count, 1)
            ep_tgt_can_long_rate = ep_tgt_can_long_sum / ep_tgt_fireable_steps
            ep_tgt_can_short_rate = ep_tgt_can_short_sum / ep_tgt_fireable_steps
            ep_tgt_fireable_rate = (ep_tgt_can_long_sum + ep_tgt_can_short_sum) / ep_tgt_fireable_steps

            episode_records.append({
                "episode": ep,
                "seed": int(self.cfg["train"].get("seed", 0) + 9999 + ep),
                "winner": winner,
                "reason": str(info.get("reason", "")),
                "steps": ep_steps,
                "return": float(ep_return),
                "red_kills": float(metrics.get("red_kills", 0)),
                "blue_kills": float(metrics.get("blue_kills", 0)),
                "red_alive": int(metrics.get("red_alive", 0)),
                "blue_alive": int(metrics.get("blue_alive", 0)),
                "red_missiles_remaining": int(ep_red_missiles),
                "blue_missiles_remaining": int(ep_blue_missiles),
                "target_action_nonzero_rate": float(ep_target_rate),
                "fire_action_nonzero_rate": float(ep_fire_rate),
                "red_fireable_edges_mean": float(ep_fireable_mean),
                "blue_fireable_edges_mean": float(ep_blue_fireable_mean),
                "red_attempted_edges_mean": float(ep_attempted_mean),
                "blue_attempted_edges_mean": float(ep_blue_attempted_mean),
                "red_selected_edges_mean": float(ep_selected_mean),
                "blue_selected_edges_mean": float(ep_blue_selected_mean),
                "red_invalid_fire_count_mean": float(ep_invalid_mean),
                "blue_invalid_fire_count_mean": float(ep_blue_invalid_mean),
                "selected_expected_exchange_mean": float(ep_sel_exch_mean),
                "missiles_launched_long": int(metrics.get("missiles_launched_long", 0)),
                "missiles_launched_short": int(metrics.get("missiles_launched_short", 0)),
                "missiles_hit": int(metrics.get("missiles_hit", 0)),
                "missiles_missed": int(metrics.get("missiles_missed", 0)),
                "fire_argmax_nonzero_rate": float(ep_fire_argmax_nz_rate),
                "fire_noop_prob_mean": float(ep_fire_noop_mean),
                "fire_long_prob_mean": float(ep_fire_long_mean),
                "fire_short_prob_mean": float(ep_fire_short_mean),
                "fire_entropy_mean": float(ep_fire_ent_mean),
                "fire_valid_mask_long_rate": float(ep_fire_mask_long_rate),
                "fire_valid_mask_short_rate": float(ep_fire_mask_short_rate),
                "adapter_zeroed_fire_rate": float(ep_adapter_zeroed_rate),
                "target_selected_fireable_rate": float(ep_tgt_fireable_rate),
                "target_selected_nonfireable_rate": 1.0 - float(ep_tgt_fireable_rate),
                "selected_target_can_long_rate": float(ep_tgt_can_long_rate),
                "selected_target_can_short_rate": float(ep_tgt_can_short_rate),
            })

            diag_missiles_long += int(metrics.get("missiles_launched_long", 0))
            diag_missiles_short += int(metrics.get("missiles_launched_short", 0))
            diag_missiles_hit += int(metrics.get("missiles_hit", 0))
            diag_missiles_missed += int(metrics.get("missiles_missed", 0))
            diag_sel_exch += float(metrics.get("selected_expected_exchange", 0.0))
            diag_blue_attempted += ep_blue_attempted
            diag_blue_selected += ep_blue_selected
            diag_blue_invalid += ep_blue_invalid
            diag_blue_fireable += ep_blue_fireable
            diag_blue_fireable_count += ep_blue_fireable_count
            diag_sel_exch_steps += ep_sel_exch_steps
            # Fire prob diag accumulation
            diag_fire_prob_no_fire += ep_fire_prob_no_fire
            diag_fire_prob_long += ep_fire_prob_long
            diag_fire_prob_short += ep_fire_prob_short
            diag_fire_entropy += ep_fire_entropy
            diag_fire_prob_steps += ep_fire_prob_steps
            diag_fire_mask_long += ep_fire_mask_long
            diag_fire_mask_short += ep_fire_mask_short
            diag_adapter_zeroed_fire += ep_adapter_zeroed_fire
            diag_fire_nonzero_count += ep_fire_nonzero_count
            diag_target_nonzero_count += ep_target_nonzero_count
            diag_tgt_can_long_sum += ep_tgt_can_long_sum
            diag_tgt_can_short_sum += ep_tgt_can_short_sum

            diag_fire_argmax_nonzero += ep_fire_argmax_nonzero
            diag_blue_fireable += int(metrics.get("blue_fireable_edges", 0))

            diag_target_nonzero_steps += ep_target_nonzero_steps
            diag_fire_nonzero_steps += ep_fire_nonzero_steps
            diag_total_steps_diag += ep_steps
            diag_attempted_edges += ep_attempted_edges
            diag_selected_edges += ep_selected_edges
            diag_invalid_fire += ep_invalid_fire
            diag_fireable_edges += ep_fireable_edges
            diag_fireable_agents += ep_fireable_agents
            diag_fireable_count += ep_fireable_count
            diag_missiles_remaining += ep_red_missiles
            diag_missile_record_count += 1

            if eval_prefix == "gui_eval":
                print(
                    f"[gui_eval_diag] red_attempted={ep_attempted_mean:.2f} "
                    f"red_selected={ep_selected_mean:.2f} "
                    f"red_invalid={ep_invalid_mean:.2f} "
                    f"red_fireable_edges_mean={ep_fireable_mean:.1f} "
                    f"target_nonzero_rate={ep_target_rate:.3f} "
                    f"fire_nonzero_rate={ep_fire_rate:.3f} "
                    f"red_missiles_remaining={ep_red_missiles}",
                    flush=True,
                )

        eval_env.engine.close()

        win_rate = red_wins / max(num_episodes, 1)
        avg_steps = total_steps / max(num_episodes, 1)
        avg_return = total_return / max(num_episodes, 1)
        avg_red_kills = total_red_kills / max(num_episodes, 1)
        avg_blue_kills = total_blue_kills / max(num_episodes, 1)

        # Overall diagnostic rates
        diag_denom = max(diag_total_steps_diag, 1)
        diag_fireable_denom = max(diag_fireable_count, 1)
        diag_candidate_denom = max(diag_candidate_pairs, 1)
        diag_missile_denom = max(diag_missile_record_count, 1)
        red_target_nonzero_rate = diag_target_nonzero_steps / diag_denom
        red_fire_nonzero_rate = diag_fire_nonzero_steps / diag_denom
        red_attempted_edges = diag_attempted_edges / diag_denom
        red_selected_edges = diag_selected_edges / diag_denom
        red_invalid_fire_count = float(diag_invalid_fire)
        red_fireable_edges = diag_fireable_edges / diag_fireable_denom
        red_fireable_agents = diag_fireable_agents / diag_fireable_denom
        red_candidate_valid_count = diag_valid_candidates / diag_denom
        red_long_available_rate = diag_long_available / diag_candidate_denom
        red_short_available_rate = diag_short_available / diag_candidate_denom
        red_missiles_remaining = diag_missiles_remaining / diag_missile_denom

        # Blue step-mean metrics
        blue_attempted_edges = diag_blue_attempted / diag_denom
        blue_selected_edges = diag_blue_selected / diag_denom
        blue_fireable_edges = diag_blue_fireable / max(diag_blue_fireable_count, 1)
        blue_invalid_fire_count = float(diag_blue_invalid)
        sel_exch_mean = diag_sel_exch / max(diag_sel_exch_steps, 1)

        # Fire head diagnostics
        fire_prob_denom = max(diag_fire_prob_steps, 1)
        fire_noop_prob_mean = diag_fire_prob_no_fire / fire_prob_denom
        fire_long_prob_mean = diag_fire_prob_long / fire_prob_denom
        fire_short_prob_mean = diag_fire_prob_short / fire_prob_denom
        fire_entropy_mean = diag_fire_entropy / fire_prob_denom
        fire_mask_long_rate = diag_fire_mask_long / fire_prob_denom
        fire_mask_short_rate = diag_fire_mask_short / fire_prob_denom
        fire_argmax_nonzero_rate = diag_fire_argmax_nonzero / fire_prob_denom

        # Adapter zeroing / target fireability diagnostics
        adapter_zeroed_rate = diag_adapter_zeroed_fire / max(diag_fire_nonzero_count, 1)
        tgt_fireable_denom = max(diag_target_nonzero_count, 1)
        target_selected_fireable_rate = (diag_tgt_can_long_sum + diag_tgt_can_short_sum) / tgt_fireable_denom
        selected_target_can_long_rate = diag_tgt_can_long_sum / tgt_fireable_denom
        selected_target_can_short_rate = diag_tgt_can_short_sum / tgt_fireable_denom

        # --- Write eval report ---
        if write_report:
            n_eps = max(num_episodes, 1)
            report_summary = {
                "env_steps": step_value,
                "update_idx": int(self.update_idx),
                "deterministic": deterministic,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "episode_len": avg_steps,
                "red_wins": float(red_wins),
                "blue_wins": float(blue_wins),
                "draws": float(draws),
                "red_kills": avg_red_kills,
                "blue_kills": avg_blue_kills,
                # Step-mean metrics (per-step averages over episodes)
                "target_action_nonzero_rate": red_target_nonzero_rate,
                "fire_action_nonzero_rate": red_fire_nonzero_rate,
                "red_fireable_edges_mean": red_fireable_edges,
                "blue_fireable_edges_mean": blue_fireable_edges,
                "red_attempted_edges_mean": red_attempted_edges,
                "blue_attempted_edges_mean": blue_attempted_edges,
                "red_selected_edges_mean": red_selected_edges,
                "blue_selected_edges_mean": blue_selected_edges,
                "red_invalid_fire_count_mean": red_invalid_fire_count,
                "blue_invalid_fire_count_mean": blue_invalid_fire_count,
                "selected_expected_exchange_mean": sel_exch_mean,
                # Episode-final cumulative metrics
                "missiles_launched_long": diag_missiles_long / n_eps,
                "missiles_launched_short": diag_missiles_short / n_eps,
                "missiles_hit": diag_missiles_hit / n_eps,
                "missiles_missed": diag_missiles_missed / n_eps,
                "red_missiles_remaining": red_missiles_remaining,
                # Fire head diagnostics
                "fire_argmax_nonzero_rate": fire_argmax_nonzero_rate,
                "fire_noop_prob_mean": fire_noop_prob_mean,
                "fire_long_prob_mean": fire_long_prob_mean,
                "fire_short_prob_mean": fire_short_prob_mean,
                "fire_entropy_mean": fire_entropy_mean,
                "fire_valid_mask_long_rate": fire_mask_long_rate,
                "fire_valid_mask_short_rate": fire_mask_short_rate,
                # Adapter zeroing / target fireability diagnostics
                "adapter_zeroed_fire_rate": adapter_zeroed_rate,
                "target_selected_fireable_rate": target_selected_fireable_rate,
                "target_selected_nonfireable_rate": 1.0 - target_selected_fireable_rate,
                "selected_target_can_long_rate": selected_target_can_long_rate,
                "selected_target_can_short_rate": selected_target_can_short_rate,
                # Backward-compat aliases
                "target_nonzero_rate": red_target_nonzero_rate,
                "fire_nonzero_rate": red_fire_nonzero_rate,
                "red_fireable_edges": red_fireable_edges,
                "red_attempted_edges": red_attempted_edges,
                "red_selected_edges": red_selected_edges,
                "red_invalid_fire_count": red_invalid_fire_count,
            }
            self._write_eval_report(
                kind=kind,
                summary=report_summary,
                episodes=episode_records,
                checkpoint_path=checkpoint_path,
            )

        print(
            f"[{eval_prefix}] done step={step_value} win_rate={win_rate:.3f} "
            f"avg_steps={avg_steps:.1f} avg_return={avg_return:.3f}",
            flush=True,
        )
        return {
            "win_rate": win_rate,
            "red_wins": float(red_wins),
            "blue_wins": float(blue_wins),
            "draws": float(draws),
            "truncated": float(truncated_count),
            "avg_steps": avg_steps,
            "avg_return": avg_return,
            "episode_len": avg_steps,
            "red_kills": avg_red_kills,
            "blue_kills": avg_blue_kills,
            # Diagnostics — bare keys for direct consumption
            "target_nonzero_rate": red_target_nonzero_rate,
            "fire_nonzero_rate": red_fire_nonzero_rate,
            "target_action_nonzero_rate": red_target_nonzero_rate,
            "fire_action_nonzero_rate": red_fire_nonzero_rate,
            "red_attempted_edges": red_attempted_edges,
            "red_selected_edges": red_selected_edges,
            "red_invalid_fire_count": red_invalid_fire_count,
            # Fire head diagnostics
            "fire_argmax_nonzero_rate": fire_argmax_nonzero_rate,
            "fire_noop_prob_mean": fire_noop_prob_mean,
            "fire_long_prob_mean": fire_long_prob_mean,
            "fire_short_prob_mean": fire_short_prob_mean,
            "fire_entropy_mean": fire_entropy_mean,
            "fire_valid_mask_long_rate": fire_mask_long_rate,
            "fire_valid_mask_short_rate": fire_mask_short_rate,
            # Adapter zeroing / target fireability diagnostics
            "adapter_zeroed_fire_rate": adapter_zeroed_rate,
            "target_selected_fireable_rate": target_selected_fireable_rate,
            "target_selected_nonfireable_rate": 1.0 - target_selected_fireable_rate,
            "selected_target_can_long_rate": selected_target_can_long_rate,
            "selected_target_can_short_rate": selected_target_can_short_rate,
            # Diagnostics — prefixed for backward compat
            "diagnostics_red_target_nonzero_rate": red_target_nonzero_rate,
            "diagnostics_red_fire_nonzero_rate": red_fire_nonzero_rate,
            "diagnostics_red_attempted_edges": red_attempted_edges,
            "diagnostics_red_selected_edges": red_selected_edges,
            "diagnostics_red_invalid_fire_count": red_invalid_fire_count,
            "diagnostics_red_fireable_edges": red_fireable_edges,
            "diagnostics_red_fireable_agents": red_fireable_agents,
            "diagnostics_red_candidate_valid_count": red_candidate_valid_count,
            "diagnostics_red_long_available_rate": red_long_available_rate,
            "diagnostics_red_short_available_rate": red_short_available_rate,
            "diagnostics_red_missiles_remaining": red_missiles_remaining,
        }
