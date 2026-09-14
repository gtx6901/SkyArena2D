"""Recurrent entity-MAPPO trainer for the SkyArena2D baseline.

The trainer owns orchestration only. Observation construction, action
distributions, recurrent chunking, credit assignment, and opponent sampling
live in dedicated modules.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from skyarena2d.training.action_adapter import SkyArenaActionAdapter

from ..adapters.skyarena_mappo_env import SkyArenaMAPPOEnv
from ..adapters.vector_env import build_env_runner
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
from .ppo_utils import (
    chunk_initial_states,
    clipped_value_loss,
    gather_recurrent_chunks,
    iter_recurrent_minibatches,
    linear_lr,
    make_recurrent_chunks,
    normalize_advantages,
    set_optimizer_lr,
)
from .rollout import (
    RolloutBatch,
    allocate_batched_obs,
    allocate_rollout_obs,
    compute_gae,
    evaluate_policy_heads,
    fill_batched_obs,
    sample_policy_actions,
)

_EVAL_EPISODE_SUM_KEYS = (
    "red_fireable_edges",
    "blue_fireable_edges",
    "fireability_edge_advantage",
    "red_fireable_agents",
    "blue_fireable_agents",
    "expected_red_kills_proxy",
    "expected_blue_kills_proxy",
    "expected_exchange_proxy",
    "red_attempted_edges",
    "blue_attempted_edges",
    "red_selected_edges",
    "blue_selected_edges",
    "selected_edge_advantage",
    "selected_expected_red_kills",
    "selected_expected_blue_kills",
    "selected_expected_exchange",
)


def _accumulate_eval_step_metrics(
    totals: dict[str, float], step_metrics: Mapping[str, Any]
) -> None:
    """Accumulate instantaneous engine metrics into episode-level eval totals."""
    for key in _EVAL_EPISODE_SUM_KEYS:
        value = step_metrics.get(key)
        if isinstance(value, (int, float, np.number)):
            totals[key] = totals.get(key, 0.0) + float(value)


class SkyArenaMAPPOTrainer:
    """CTDE MAPPO with an entity-set recurrent actor and per-agent critic."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.train_cfg = cfg["train"]
        self.env_cfg = cfg["env"]
        self.model_cfg = cfg["model"]
        self.logging_cfg = cfg.get("logging", {})
        self.eval_cfg = cfg.get("evaluation", {})

        self.num_envs = int(self.train_cfg.get("num_envs", 16))
        self.rollout_steps = int(self.train_cfg.get("rollout_steps", 128))
        self.total_env_steps = int(self.train_cfg.get("total_env_steps", 5_000_000))
        self.ppo_epochs = int(self.train_cfg.get("ppo_epochs", 4))
        self.gamma = float(self.train_cfg.get("gamma", 0.99))
        self.gae_lambda = float(self.train_cfg.get("gae_lambda", 0.95))
        self.clip_coef = float(self.train_cfg.get("clip_coef", 0.2))
        self.value_clip_coef = float(self.train_cfg.get("value_clip_coef", self.clip_coef))
        self.entropy_coef = float(self.train_cfg.get("entropy_coef", 0.01))
        self.value_coef = float(self.train_cfg.get("value_coef", 0.5))
        self.max_grad_norm = float(self.train_cfg.get("max_grad_norm", 0.5))
        self.recurrent_chunk_length = int(self.train_cfg.get("recurrent_chunk_length", 32))
        self.num_minibatches = int(self.train_cfg.get("num_minibatches", 8))
        self.target_kl = float(self.train_cfg.get("target_kl", 0.0))
        self.initial_lr = float(self.train_cfg.get("learning_rate", 3e-4))
        self.min_lr = float(self.train_cfg.get("min_learning_rate", 0.0))
        self.use_lr_decay = str(self.train_cfg.get("lr_decay", "linear")).lower() == "linear"
        self.use_popart = bool(self.train_cfg.get("popart", True))
        self.team_reward_start = float(self.train_cfg.get("team_reward_start", 0.5))
        self.team_reward_end = float(self.train_cfg.get("team_reward_end", 1.0))
        self.reward_mix_steps = int(self.train_cfg.get("reward_mix_steps", 1_000_000))
        self.save_interval = int(self.train_cfg.get("save_interval", 100_000))
        self.eval_interval = int(
            self.eval_cfg.get("policy_eval_interval", self.train_cfg.get("eval_interval", 50_000))
        )
        self.eval_episodes = int(self.eval_cfg.get("policy_eval_episodes", 10))

        self.env_runner = build_env_runner(cfg, self.num_envs)
        self.opponent_pool = self.env_runner.opponent_pool
        self.num_agents = self.env_runner.num_agents
        obs_shapes = self.env_runner.obs_shapes
        self.entity_slots = obs_shapes["entity_features"][1]

        requested_device = str(self.train_cfg.get("device", "cpu"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print("[device] CUDA unavailable; falling back to CPU", flush=True)
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        self.actor = SkyArenaActor(
            self_dim=obs_shapes["self_features"][-1],
            entity_dim=obs_shapes["entity_features"][-1],
            candidate_slots=self.entity_slots,
            num_agents=self.num_agents,
            trunk_dim=int(self.model_cfg.get("trunk_dim", 192)),
            lstm_hidden_dim=int(self.model_cfg.get("lstm_hidden_dim", 192)),
            entity_embed_dim=int(self.model_cfg.get("entity_embed_dim", 128)),
            course_bins=9,
            agent_id_embed_dim=int(self.model_cfg.get("agent_id_embed_dim", 16)),
            attention_heads=int(self.model_cfg.get("attention_heads", 4)),
            attention_layers=int(self.model_cfg.get("attention_layers", 2)),
        ).to(self.device)
        self.critic = SkyArenaCritic(
            global_state_dim=obs_shapes["global_state"][0],
            hidden_dim=int(self.model_cfg.get("critic_hidden_dim", 256)),
            num_agents=self.num_agents,
            agent_id_embed_dim=int(self.model_cfg.get("agent_id_embed_dim", 16)),
            use_popart=self.use_popart,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters()], lr=self.initial_lr
        )

        engine_cfg = self.env_runner.engine_config
        self.action_adapter = SkyArenaActionAdapter(
            candidate_slots=self.entity_slots,
            course_bins=9,
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

        run_cfg = dict(self.train_cfg)
        run_cfg["logging"] = self.logging_cfg
        self.run_cfg = run_cfg
        self.run_dirs = ensure_run_dirs(run_cfg)
        save_run_config(self.run_dirs, cfg)
        self.env_steps = 0
        self.update_idx = 0
        self._load_requested_checkpoint()

        self.current_obs = self.env_runner.reset_all()
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        hidden_dim = self.actor.lstm_hidden_dim
        self.actor_h = torch.zeros(
            self.num_envs, self.num_agents, hidden_dim, device=self.device
        )
        self.actor_c = torch.zeros_like(self.actor_h)
        self.writer = build_writer(run_cfg, purge_step=self.env_steps or None)
        self.next_save_step = self.env_steps + self.save_interval
        self.next_eval_step = self.env_steps + self.eval_interval

    def _load_requested_checkpoint(self) -> None:
        init_path = str(self.train_cfg.get("init_checkpoint", "")).strip()
        checkpoint: Path | str | None = None
        resume = bool(self.train_cfg.get("resume", False))
        if resume:
            checkpoint = latest_checkpoint(self.run_cfg)
        elif init_path:
            checkpoint = init_path
        if checkpoint is None:
            return
        payload = load_checkpoint(
            checkpoint,
            self.actor,
            self.critic,
            self.optimizer if resume else None,
            map_location=self.device,
        )
        if resume:
            self.env_steps = int(payload.get("env_steps", 0))
            self.update_idx = int(payload.get("update_idx", 0))
            if self.opponent_pool is not None and "opponent_pool" in payload:
                self.opponent_pool.load_state_dict(payload["opponent_pool"])
        print(f"[checkpoint] loaded {checkpoint} at env_steps={self.env_steps}", flush=True)

    @staticmethod
    def _reset_recurrent_hidden_after_done(
        h: torch.Tensor,
        c: torch.Tensor,
        prev_done: np.ndarray,
        num_agents: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del num_agents
        if not np.any(prev_done):
            return h, c
        if h.ndim == 2:
            reset = torch.as_tensor(prev_done, dtype=h.dtype, device=h.device).reshape(-1, 1)
            reset = reset.repeat_interleave(c.shape[0] // max(reset.shape[0], 1), dim=0)
        else:
            reset = torch.as_tensor(prev_done, dtype=h.dtype, device=h.device).reshape(-1, 1, 1)
        return h * (1.0 - reset), c * (1.0 - reset)

    def _team_weight(self) -> float:
        if self.reward_mix_steps <= 0:
            return self.team_reward_end
        fraction = min(max(self.env_steps / self.reward_mix_steps, 0.0), 1.0)
        return self.team_reward_start + fraction * (
            self.team_reward_end - self.team_reward_start
        )

    def _decode_action(self, env_index: int, obs: Mapping[str, np.ndarray], sampled: dict):
        context = self.env_runner.action_contexts[env_index]
        return self.action_adapter.decode(
            course_action=sampled["course"][env_index],
            target_action=sampled["target"][env_index],
            fire_action=sampled["fire"][env_index],
            own=context,
            candidate_ids=obs["candidate_ids"],
            candidate_can_long=obs["candidate_can_long"],
            candidate_can_short=obs["candidate_can_short"],
            has_active_contact=obs["has_active_contact"] > 0.5,
            current_heading=context.current_heading,
            entity_features=obs["entity_features"],
            ew_state_key=env_index,
            step_count=context.step_count,
        )

    def _step_all(self, actions: list) -> list[tuple[dict, float, bool, dict]]:
        return self.env_runner.step_all(actions)

    @staticmethod
    def _numeric_means(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for row in rows:
            for key, value in row.items():
                if isinstance(value, (bool, int, float, np.integer, np.floating)):
                    buckets.setdefault(key, []).append(float(value))
        return {key: float(np.mean(values)) for key, values in buckets.items() if values}

    def _collect_rollout(self) -> tuple[RolloutBatch, list[dict]]:
        example = self.current_obs[0]
        obs_batch = allocate_batched_obs(example, self.num_envs)
        observations = allocate_rollout_obs(example, self.rollout_steps, self.num_envs)
        shape = (self.rollout_steps, self.num_envs, self.num_agents)
        recurrent_shape = (*shape, self.actor.lstm_hidden_dim)
        recurrent_h_device = torch.empty(
            recurrent_shape, dtype=torch.float32, device=self.device
        )
        recurrent_c_device = torch.empty_like(recurrent_h_device)
        course = np.empty(shape, dtype=np.int64)
        target = np.empty(shape, dtype=np.int64)
        fire = np.empty(shape, dtype=np.int64)
        log_prob = np.empty(shape, dtype=np.float32)
        reward = np.empty(shape, dtype=np.float32)
        done = np.empty(shape[:2], dtype=bool)
        value = np.empty(shape, dtype=np.float32)
        episode_stats: list[dict] = []
        step_metrics: list[dict] = []
        policy_seconds = 0.0
        value_seconds = 0.0
        decode_seconds = 0.0
        env_step_seconds = 0.0
        reset_seconds = 0.0

        for step in range(self.rollout_steps):
            fill_batched_obs(obs_batch, self.current_obs)
            for key in observations:
                observations[key][step] = obs_batch[key]
            recurrent_h_device[step].copy_(self.actor_h)
            recurrent_c_device[step].copy_(self.actor_c)

            phase_start = time.perf_counter()
            with torch.no_grad():
                sampled = sample_policy_actions(
                    self.actor,
                    obs_batch,
                    (self.actor_h, self.actor_c),
                    self.device,
                    deterministic=False,
                )
            policy_seconds += time.perf_counter() - phase_start

            course[step] = sampled["course"]
            target[step] = sampled["target"]
            fire[step] = sampled["fire"]
            log_prob[step] = sampled["log_prob"]
            phase_start = time.perf_counter()
            actions = [
                self._decode_action(index, self.current_obs[index], sampled)
                for index in range(self.num_envs)
            ]
            decode_seconds += time.perf_counter() - phase_start
            phase_start = time.perf_counter()
            results = self._step_all(actions)
            env_step_seconds += time.perf_counter() - phase_start
            next_obs: list[dict] = []
            reset_indices: list[int] = []
            team_weight = self._team_weight()
            done_step = np.zeros(self.num_envs, dtype=bool)
            for env_index, (new_obs, team_reward, env_done, info) in enumerate(results):
                agent_reward = np.asarray(
                    info.get("red_agent_reward", np.zeros(self.num_agents)), dtype=np.float32
                )
                if agent_reward.shape != (self.num_agents,):
                    raise ValueError(
                        f"red_agent_reward must have shape {(self.num_agents,)}, got {agent_reward.shape}"
                    )
                reward[step, env_index] = (
                    (1.0 - team_weight) * agent_reward + team_weight * float(team_reward)
                )
                done_step[env_index] = env_done
                self._episode_returns[env_index] += float(team_reward)
                self._episode_lengths[env_index] += 1
                metrics = info.get("metrics", {})
                if isinstance(metrics, dict):
                    step_metrics.append(metrics)
                if env_done:
                    episode_stats.append({
                        "winner": str(info.get("winner", "draw")),
                        "episode_return": float(self._episode_returns[env_index]),
                        "episode_len": int(self._episode_lengths[env_index]),
                        "opponent_name": str(info.get("opponent_name", "unknown")),
                        **(metrics if isinstance(metrics, dict) else {}),
                    })
                    self._episode_returns[env_index] = 0.0
                    self._episode_lengths[env_index] = 0
                    self.action_adapter.reset_ew_state(env_index)
                    reset_indices.append(env_index)
                next_obs.append(new_obs)

            if reset_indices:
                phase_start = time.perf_counter()
                reset_observations = self.env_runner.reset_many(reset_indices)
                reset_seconds += time.perf_counter() - phase_start
                for env_index, new_obs in reset_observations.items():
                    next_obs[env_index] = new_obs

            done[step] = done_step
            self.current_obs = next_obs
            self.actor_h = sampled["next_h"].to(self.device)
            self.actor_c = sampled["next_c"].to(self.device)
            self.actor_h, self.actor_c = self._reset_recurrent_hidden_after_done(
                self.actor_h, self.actor_c, done_step, self.num_agents
            )
            self.env_steps += self.num_envs

        fill_batched_obs(obs_batch, self.current_obs)
        phase_start = time.perf_counter()
        with torch.no_grad():
            rollout_states = observations["global_state"]
            state_dim = rollout_states.shape[-1]
            all_states = np.concatenate(
                (
                    rollout_states.reshape(-1, state_dim),
                    obs_batch["global_state"],
                ),
                axis=0,
            )
            all_values = self.critic(torch.as_tensor(
                all_states, dtype=torch.float32, device=self.device
            )).cpu().numpy()
        rollout_value_count = self.rollout_steps * self.num_envs
        value[:] = all_values[:rollout_value_count].reshape(value.shape)
        next_value = all_values[rollout_value_count:]
        value_seconds += time.perf_counter() - phase_start
        recurrent_h = recurrent_h_device.cpu().numpy()
        recurrent_c = recurrent_c_device.cpu().numpy()
        batch = RolloutBatch(
            observations=observations,
            recurrent_h=recurrent_h,
            recurrent_c=recurrent_c,
            course_action=course,
            target_action=target,
            fire_action=fire,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value,
            next_value=next_value,
            episode_stats=episode_stats,
        )
        batch.validate()
        self._last_collect_timing = {
            "policy_seconds": policy_seconds,
            "value_seconds": value_seconds,
            "action_decode_seconds": decode_seconds,
            "env_step_seconds": env_step_seconds,
            "env_reset_seconds": reset_seconds,
        }
        return batch, step_metrics

    @staticmethod
    def _pack_time_major(array: np.ndarray, chunks) -> tuple[np.ndarray, np.ndarray]:
        packed, valid = gather_recurrent_chunks(array, chunks)
        axes = (1, 0, *range(2, packed.ndim))
        return packed.transpose(axes), valid.T

    def _ppo_update(self, batch: RolloutBatch) -> dict[str, float]:
        advantages, returns = compute_gae(
            batch.reward,
            batch.value,
            batch.next_value,
            batch.done,
            self.gamma,
            self.gae_lambda,
        )
        live_np = batch.observations["alive_mask"].astype(bool)
        advantages = normalize_advantages(
            torch.as_tensor(advantages, device=self.device),
            torch.as_tensor(live_np, device=self.device),
        ).cpu().numpy()
        if self.use_popart:
            self.critic.update_popart(torch.as_tensor(returns[live_np], device=self.device))

        chunks = make_recurrent_chunks(
            batch.done, self.num_agents, self.recurrent_chunk_length
        )
        minibatch_chunks = max(1, math.ceil(len(chunks) / max(self.num_minibatches, 1)))
        if self.use_lr_decay:
            learning_rate = linear_lr(
                self.initial_lr,
                self.env_steps,
                self.total_env_steps,
                final_lr=self.min_lr,
            )
            set_optimizer_lr(self.optimizer, learning_rate)
        else:
            learning_rate = self.initial_lr

        global_by_agent = np.broadcast_to(
            batch.observations["global_state"][:, :, None, :],
            (*batch.value.shape, batch.observations["global_state"].shape[-1]),
        )
        totals = {
            "loss": 0.0,
            "pg_loss": 0.0,
            "vf_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        updates = 0
        stop_early = False
        rng = np.random.default_rng(int(self.train_cfg.get("seed", 0)) + self.update_idx)

        for _epoch in range(self.ppo_epochs):
            for chunk_group in iter_recurrent_minibatches(
                chunks, minibatch_chunks, shuffle=True, rng=rng
            ):
                actor_batch: dict[str, torch.Tensor] = {}
                for key in ("self_features", "entity_features", "entity_mask", "agent_id"):
                    packed, _ = self._pack_time_major(batch.observations[key], chunk_group)
                    dtype = torch.long if key == "agent_id" else (
                        torch.bool if key == "entity_mask" else torch.float32
                    )
                    actor_batch[key] = torch.as_tensor(packed, dtype=dtype, device=self.device)
                h0_np, c0_np = chunk_initial_states(
                    batch.recurrent_h, batch.recurrent_c, chunk_group
                )
                outputs = self.actor.forward_sequence(
                    actor_batch,
                    (
                        torch.as_tensor(h0_np, dtype=torch.float32, device=self.device),
                        torch.as_tensor(c0_np, dtype=torch.float32, device=self.device),
                    ),
                    episode_starts=None,
                )

                values: dict[str, torch.Tensor] = {}
                valid_np: np.ndarray | None = None
                sources = {
                    "course": batch.course_action,
                    "target": batch.target_action,
                    "fire": batch.fire_action,
                    "old_log_prob": batch.log_prob,
                    "old_value": batch.value,
                    "advantage": advantages,
                    "returns": returns,
                    "alive": live_np,
                    "entity_mask": batch.observations["entity_mask"],
                    "target_mask": batch.observations["target_mask"],
                    "can_long": batch.observations["candidate_can_long"],
                    "can_short": batch.observations["candidate_can_short"],
                    "global_state": global_by_agent,
                    "agent_id": batch.observations["agent_id"],
                }
                for key, source in sources.items():
                    packed, valid_now = self._pack_time_major(source, chunk_group)
                    valid_np = valid_now if valid_np is None else valid_np
                    if packed.dtype == np.bool_:
                        dtype = torch.bool
                    elif np.issubdtype(packed.dtype, np.integer):
                        dtype = torch.long
                    else:
                        dtype = torch.float32
                    values[key] = torch.as_tensor(packed, dtype=dtype, device=self.device)
                assert valid_np is not None
                valid = torch.as_tensor(valid_np, dtype=torch.bool, device=self.device)
                loss_mask = valid & values["alive"].bool()

                evaluated = evaluate_policy_heads(
                    course_logits=outputs["course_logits"],
                    target_logits=outputs["target_logits"],
                    fire_logits=outputs["fire_logits"],
                    course_action=values["course"],
                    target_action=values["target"],
                    fire_action=values["fire"],
                    entity_mask=values["entity_mask"],
                    alive_mask=values["alive"],
                    candidate_can_long=values["can_long"],
                    candidate_can_short=values["can_short"],
                    target_mask=values["target_mask"],
                )
                new_log_prob = evaluated["log_prob"]
                old_log_prob = values["old_log_prob"]
                log_ratio = new_log_prob - old_log_prob
                ratio = log_ratio.exp()
                pg_unclipped = -values["advantage"] * ratio
                pg_clipped = -values["advantage"] * ratio.clamp(
                    1.0 - self.clip_coef, 1.0 + self.clip_coef
                )
                weights = loss_mask.to(torch.float32)
                denominator = weights.sum().clamp_min(1.0)
                pg_loss = (
                    torch.maximum(pg_unclipped, pg_clipped) * weights
                ).sum() / denominator
                entropy = (evaluated["entropy"] * weights).sum() / denominator

                predicted_value = self.critic(
                    values["global_state"],
                    values["agent_id"],
                    normalized=self.use_popart,
                )
                value_target = values["returns"]
                old_value = values["old_value"]
                if self.use_popart:
                    value_target = self.critic.normalize_targets(value_target)
                    old_value = self.critic.normalize_targets(old_value)
                value_loss = clipped_value_loss(
                    predicted_value,
                    old_value,
                    value_target,
                    self.value_clip_coef,
                    valid_mask=loss_mask,
                )
                loss = (
                    pg_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*self.actor.parameters(), *self.critic.parameters()],
                    self.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (
                        ((ratio - 1.0) - log_ratio) * weights
                    ).sum() / denominator
                    clip_fraction = (
                        ((ratio - 1.0).abs() > self.clip_coef).float() * weights
                    ).sum() / denominator
                for key, item in (
                    ("loss", loss),
                    ("pg_loss", pg_loss),
                    ("vf_loss", value_loss),
                    ("entropy", entropy),
                    ("approx_kl", approx_kl),
                    ("clip_fraction", clip_fraction),
                ):
                    totals[key] += float(item.detach().cpu())
                updates += 1
                if self.target_kl > 0.0 and float(approx_kl) > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        self.update_idx += 1
        result = {key: value / max(updates, 1) for key, value in totals.items()}
        result.update({
            "learning_rate": learning_rate,
            "team_reward_weight": self._team_weight(),
            "recurrent_chunks": float(len(chunks)),
            "ppo_minibatches": float(updates),
            "early_stop_kl": float(stop_early),
        })
        return result

    def _rollout_diagnostics(self, batch: RolloutBatch) -> dict[str, float]:
        alive = batch.observations["alive_mask"].astype(bool)
        can_fire = (
            batch.observations["candidate_can_long"]
            | batch.observations["candidate_can_short"]
        )
        live_count = max(int(alive.sum()), 1)
        return {
            "target_action_nonzero_rate": float(
                ((batch.target_action > 0) & alive).sum() / live_count
            ),
            "fire_action_nonzero_rate": float(
                ((batch.fire_action > 0) & alive).sum() / live_count
            ),
            "fireable_agent_rate": float(
                (can_fire.any(axis=-1) & alive).sum() / live_count
            ),
            "alive_rate": float(alive.mean()),
            "reward_mean": float(batch.reward.mean()),
        }

    def _save(self, *, final: bool = False) -> Path:
        extra = {}
        if self.opponent_pool is not None:
            extra["opponent_pool"] = self.opponent_pool.state_dict()
        path = save_checkpoint(
            train_cfg=self.run_cfg,
            actor=self.actor,
            critic=self.critic,
            optimizer=self.optimizer,
            env_steps=self.env_steps,
            update_idx=self.update_idx,
            extra=extra,
        )
        print(f"[checkpoint] saved{' final' if final else ''} {path}", flush=True)
        return path

    def train(self) -> None:
        print(
            f"[train] entity-MAPPO start envs={self.num_envs} "
            f"rollout={self.rollout_steps} target_steps={self.total_env_steps} "
            f"env_backend={self.env_runner.backend} "
            f"env_workers={self.env_runner.num_workers}",
            flush=True,
        )
        try:
            while self.env_steps < self.total_env_steps:
                start = time.perf_counter()
                batch, step_metrics = self._collect_rollout()
                collect_seconds = time.perf_counter() - start
                update_start = time.perf_counter()
                update_metrics = self._ppo_update(batch)
                update_seconds = time.perf_counter() - update_start
                throughput = (
                    self.num_envs * self.rollout_steps / max(collect_seconds, 1e-9)
                )
                update_metrics["sample_steps_per_second"] = throughput
                update_metrics["wall_steps_per_second"] = (
                    self.num_envs * self.rollout_steps
                    / max(collect_seconds + update_seconds, 1e-9)
                )
                update_metrics["collect_seconds"] = collect_seconds
                update_metrics["ppo_update_seconds"] = update_seconds
                update_metrics.update(self._last_collect_timing)
                log_scalars(self.writer, "train", update_metrics, self.env_steps)
                log_scalars(
                    self.writer, "rollout", self._rollout_diagnostics(batch), self.env_steps
                )
                log_scalars(
                    self.writer,
                    "environment",
                    self._numeric_means(step_metrics),
                    self.env_steps,
                )
                if batch.episode_stats:
                    winners = [row["winner"] for row in batch.episode_stats]
                    episode_summary = {
                        "win_rate": winners.count("red") / len(winners),
                        "draw_rate": winners.count("draw") / len(winners),
                        "return": float(np.mean([
                            row["episode_return"] for row in batch.episode_stats
                        ])),
                        "episode_len": float(np.mean([
                            row["episode_len"] for row in batch.episode_stats
                        ])),
                    }
                    log_scalars(self.writer, "episode", episode_summary, self.env_steps)
                print(
                    f"[train] steps={self.env_steps} loss={update_metrics['loss']:.4f} "
                    f"pg={update_metrics['pg_loss']:.4f} "
                    f"vf={update_metrics['vf_loss']:.4f} "
                    f"entropy={update_metrics['entropy']:.4f} "
                    f"sample_sps={throughput:.1f} "
                    f"wall_sps={update_metrics['wall_steps_per_second']:.1f} "
                    f"policy_s={update_metrics['policy_seconds']:.2f} "
                    f"env_s={update_metrics['env_step_seconds']:.2f} "
                    f"value_s={update_metrics['value_seconds']:.2f} "
                    f"ppo_s={update_seconds:.2f}",
                    flush=True,
                )
                if self.save_interval > 0 and self.env_steps >= self.next_save_step:
                    self._save()
                    self.next_save_step += self.save_interval
                if self.eval_interval > 0 and self.env_steps >= self.next_eval_step:
                    log_scalars(
                        self.writer,
                        "eval",
                        self.evaluate(self.eval_episodes, write_report=True),
                        self.env_steps,
                    )
                    self.next_eval_step += self.eval_interval
            self._save(final=True)
        finally:
            self.close()

    def _write_eval_report(self, kind: str, summary: dict, episodes: list[dict]) -> Path:
        report_dir = self.run_dirs["exp_dir"] / "eval_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"{kind}_step_{self.env_steps:09d}.json"
        report = {
            "kind": kind,
            "env_steps": self.env_steps,
            "update_idx": self.update_idx,
            "config": {
                "experiment_name": self.train_cfg.get("experiment_name", ""),
                "blue_rule": self.env_cfg.get("blue_rule", "fix_rule_v2"),
            },
            "summary": summary,
            "episodes": episodes,
            "timestamp_unix": time.time(),
        }
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[eval_report] wrote {path}", flush=True)
        return path

    def _build_eval_adapter(self, engine_cfg, candidate_slots: int) -> SkyArenaActionAdapter:
        return SkyArenaActionAdapter(
            candidate_slots=candidate_slots,
            course_bins=9,
            map_width=engine_cfg.map.width,
            map_height=engine_cfg.map.height,
            radar_freq_count=self.env_cfg.get(
                "radar_freq_count", engine_cfg.radar.freq_count
            ),
            use_jammer_strategy=self.env_cfg.get("use_jammer_strategy", True),
            jammer_range=self.env_cfg.get("jammer_range", engine_cfg.jamming.range),
            max_jammers_per_side=self.env_cfg.get("max_jammers_per_side", 3),
        )

    @staticmethod
    def _parallel_eval_actions(
        adapter,
        eval_runner,
        observations,
        active,
        sampled,
        target_nonzero,
        fire_nonzero,
    ) -> list:
        actions = []
        for local_index, episode in enumerate(active):
            obs = observations[episode]
            context = eval_runner.action_contexts[episode]
            actions.append(adapter.decode(
                course_action=sampled["course"][local_index],
                target_action=sampled["target"][local_index],
                fire_action=sampled["fire"][local_index],
                own=context,
                candidate_ids=obs["candidate_ids"],
                candidate_can_long=obs["candidate_can_long"],
                candidate_can_short=obs["candidate_can_short"],
                has_active_contact=obs["has_active_contact"] > 0.5,
                current_heading=context.current_heading,
                entity_features=obs["entity_features"],
                ew_state_key=episode,
                step_count=context.step_count,
            ))
            alive = obs["alive_mask"] > 0.5
            target_nonzero[episode] += np.count_nonzero(
                (sampled["target"][local_index] > 0) & alive
            )
            fire_nonzero[episode] += np.count_nonzero(
                (sampled["fire"][local_index] > 0) & alive
            )
        return actions

    @staticmethod
    def _finalize_parallel_eval_records(
        eval_runner,
        episode_returns,
        steps,
        target_nonzero,
        fire_nonzero,
        metric_totals,
        final_info,
    ) -> list[dict]:
        records = []
        for episode, info in enumerate(final_info):
            final_metrics = (
                dict(info.get("metrics", {}))
                if isinstance(info.get("metrics"), dict)
                else {}
            )
            final_metrics.update(metric_totals[episode])
            records.append({
                "episode": episode,
                "seed": eval_runner.last_reset_seeds[episode],
                "winner": str(info.get("winner", "draw")),
                "episode_return": float(episode_returns[episode]),
                "episode_len": int(steps[episode]),
                "target_action_nonzero_count": int(target_nonzero[episode]),
                "fire_action_nonzero_count": int(fire_nonzero[episode]),
                **final_metrics,
            })
        return records

    def _parallel_eval_records(
        self,
        eval_cfg: dict,
        num_episodes: int,
        *,
        deterministic: bool,
        deterministic_reset: bool,
        max_steps: int | None,
    ) -> list[dict]:
        eval_workers = min(
            num_episodes,
            int(self.eval_cfg.get("policy_eval_workers", num_episodes)),
        )
        runner_cfg = {
            **eval_cfg,
            "train": {
                **self.train_cfg,
                "env_backend": "subprocess",
                "env_workers": eval_workers,
            },
        }
        eval_runner = build_env_runner(
            runner_cfg,
            num_episodes,
            seed_offsets=[9999 + episode for episode in range(num_episodes)],
            deterministic_reset=deterministic_reset,
        )
        adapter = self._build_eval_adapter(
            eval_runner.engine_config,
            eval_runner.obs_shapes["entity_features"][1],
        )
        observations = eval_runner.reset_all()
        h = torch.zeros(
            num_episodes,
            self.num_agents,
            self.actor.lstm_hidden_dim,
            device=self.device,
        )
        c = torch.zeros_like(h)
        episode_returns = np.zeros(num_episodes, dtype=np.float64)
        steps = np.zeros(num_episodes, dtype=np.int32)
        target_nonzero = np.zeros(num_episodes, dtype=np.int64)
        fire_nonzero = np.zeros(num_episodes, dtype=np.int64)
        metric_totals = [{} for _ in range(num_episodes)]
        final_info: list[dict[str, Any]] = [{} for _ in range(num_episodes)]
        active = list(range(num_episodes))
        try:
            while active:
                if max_steps is not None:
                    reached_limit = [index for index in active if steps[index] >= max_steps]
                    for index in reached_limit:
                        active.remove(index)
                    if not active:
                        break
                obs_batch = {
                    key: np.stack([observations[index][key] for index in active])
                    for key in observations[active[0]]
                }
                sampled = sample_policy_actions(
                    self.actor,
                    obs_batch,
                    (h[active], c[active]),
                    self.device,
                    deterministic,
                )
                actions = self._parallel_eval_actions(
                    adapter,
                    eval_runner,
                    observations,
                    active,
                    sampled,
                    target_nonzero,
                    fire_nonzero,
                )
                results = eval_runner.step_many(active, actions)
                next_active = []
                for local_index, episode in enumerate(active):
                    obs, team_reward, done, info = results[episode]
                    observations[episode] = obs
                    episode_returns[episode] += float(team_reward)
                    steps[episode] += 1
                    h[episode].copy_(sampled["next_h"][local_index])
                    c[episode].copy_(sampled["next_c"][local_index])
                    step_metrics = info.get("metrics", {})
                    if isinstance(step_metrics, Mapping):
                        _accumulate_eval_step_metrics(
                            metric_totals[episode], step_metrics
                        )
                    final_info[episode] = info
                    hit_limit = max_steps is not None and steps[episode] >= max_steps
                    if not done and not hit_limit:
                        next_active.append(episode)
                active = next_active
        finally:
            eval_runner.close()

        return self._finalize_parallel_eval_records(
            eval_runner,
            episode_returns,
            steps,
            target_nonzero,
            fire_nonzero,
            metric_totals,
            final_info,
        )

    def evaluate(
        self,
        num_episodes: int = 10,
        checkpoint_path: str | None = None,
        *,
        deterministic: bool = True,
        deterministic_reset: bool = True,
        render_mode: str | None = None,
        max_steps: int | None = None,
        output_dir: Path | None = None,
        save_visual: bool = False,
        render_every: int = 1,
        step_tag: int | None = None,
        write_report: bool = False,
        kind: str = "eval",
    ) -> dict[str, float]:
        """Evaluate only against the configured anchor rule, never the pool."""
        del output_dir, save_visual, step_tag
        if checkpoint_path is not None:
            load_checkpoint(
                checkpoint_path, self.actor, self.critic, map_location=self.device
            )
        eval_cfg = {**self.cfg, "env": {**self.env_cfg, "opponent_pool": []}}
        evaluation_start = time.perf_counter()
        if render_mode is None and deterministic_reset and int(num_episodes) > 1:
            was_training = self.actor.training
            self.actor.eval()
            try:
                records = self._parallel_eval_records(
                    eval_cfg,
                    int(num_episodes),
                    deterministic=deterministic,
                    deterministic_reset=deterministic_reset,
                    max_steps=max_steps,
                )
            finally:
                self.actor.train(was_training)
        else:
            records = []
        eval_env = SkyArenaMAPPOEnv(
            eval_cfg, seed_offset=9999, deterministic_reset=deterministic_reset
        )
        adapter = self._build_eval_adapter(
            eval_env.engine_config,
            eval_env.obs_builder.entity_slots,
        )
        render_every = max(int(render_every), 1)
        was_training = self.actor.training
        self.actor.eval()
        try:
            for episode in range(0 if records else int(num_episodes)):
                obs = eval_env.reset()
                h = torch.zeros(
                    1, self.num_agents, self.actor.lstm_hidden_dim, device=self.device
                )
                c = torch.zeros_like(h)
                done = False
                episode_return = 0.0
                steps = 0
                target_nonzero = 0
                fire_nonzero = 0
                episode_metric_totals: dict[str, float] = {}
                info: dict[str, Any] = {}
                while not done and (max_steps is None or steps < max_steps):
                    obs_batch = {key: value[None] for key, value in obs.items()}
                    sampled = sample_policy_actions(
                        self.actor,
                        obs_batch,
                        (h, c),
                        self.device,
                        deterministic,
                    )
                    action = adapter.decode(
                        course_action=sampled["course"][0],
                        target_action=sampled["target"][0],
                        fire_action=sampled["fire"][0],
                        own=eval_env.engine.state.red,
                        candidate_ids=obs["candidate_ids"],
                        candidate_can_long=obs["candidate_can_long"],
                        candidate_can_short=obs["candidate_can_short"],
                        has_active_contact=obs["has_active_contact"] > 0.5,
                        current_heading=eval_env.engine.state.red.heading[: self.num_agents],
                        entity_features=obs["entity_features"],
                        ew_state_key="eval",
                        step_count=eval_env.engine.state.step_count,
                    )
                    alive = obs["alive_mask"] > 0.5
                    target_nonzero += int(np.count_nonzero(
                        (sampled["target"][0] > 0) & alive
                    ))
                    fire_nonzero += int(np.count_nonzero(
                        (sampled["fire"][0] > 0) & alive
                    ))
                    obs, team_reward, done, info = eval_env.step(action)
                    step_metrics = info.get("metrics", {})
                    if isinstance(step_metrics, Mapping):
                        _accumulate_eval_step_metrics(episode_metric_totals, step_metrics)
                    episode_return += float(team_reward)
                    h, c = sampled["next_h"], sampled["next_c"]
                    steps += 1
                    if render_mode is not None and steps % render_every == 0:
                        eval_env.engine.render(render_mode)
                final_metrics = (
                    dict(info.get("metrics", {}))
                    if isinstance(info.get("metrics"), dict)
                    else {}
                )
                final_metrics.update(episode_metric_totals)
                records.append({
                    "episode": episode,
                    "seed": eval_env.last_reset_seed,
                    "winner": str(info.get("winner", "draw")),
                    "episode_return": episode_return,
                    "episode_len": steps,
                    "target_action_nonzero_count": target_nonzero,
                    "fire_action_nonzero_count": fire_nonzero,
                    **final_metrics,
                })
        finally:
            self.actor.train(was_training)
            eval_env.engine.close()

        winners = [record["winner"] for record in records]
        denominator = max(len(winners), 1)
        summary = {
            "win_rate": winners.count("red") / denominator,
            "blue_win_rate": winners.count("blue") / denominator,
            "draw_rate": winners.count("draw") / denominator,
            "avg_return": float(np.mean([
                record["episode_return"] for record in records
            ])) if records else 0.0,
            "avg_episode_len": float(np.mean([
                record["episode_len"] for record in records
            ])) if records else 0.0,
            "evaluation_seconds": time.perf_counter() - evaluation_start,
        }
        summary["steps_per_second"] = float(
            sum(record["episode_len"] for record in records)
            / max(summary["evaluation_seconds"], 1e-9)
        )
        final_metrics = self._numeric_means(records)
        for key in (
            "red_fireable_edges",
            "red_attempted_edges",
            "red_selected_edges",
            "red_invalid_fire_count",
            "red_missiles_remaining",
            "selected_expected_exchange",
        ):
            if key in final_metrics:
                summary[key] = final_metrics[key]
        if write_report:
            self._write_eval_report(kind, summary, records)
        print(
            f"[eval] episodes={len(records)} win_rate={summary['win_rate']:.3f} "
            f"avg_len={summary['avg_episode_len']:.1f} "
            f"eval_sps={summary['steps_per_second']:.1f}",
            flush=True,
        )
        return summary

    def close(self) -> None:
        self.env_runner.close()
        if self.writer is not None:
            self.writer.close()
            self.writer = None
