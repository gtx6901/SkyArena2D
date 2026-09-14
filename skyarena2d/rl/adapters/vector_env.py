"""Synchronous serial and multi-process runners for MAPPO environments."""
from __future__ import annotations

import copy
import multiprocessing as mp
import os
import traceback
from collections.abc import Sequence
from typing import Any

from .skyarena_mappo_env import (
    SkyArenaActionContext,
    SkyArenaMAPPOEnv,
)

EnvStep = tuple[dict, float, bool, dict]


def _worker_main(connection, cfg: dict, env_indices: list[int]) -> None:
    """Own a fixed group of environments for the life of one worker."""
    envs: dict[int, SkyArenaMAPPOEnv] = {}
    try:
        envs = {
            index: SkyArenaMAPPOEnv(cfg, seed_offset=index)
            for index in env_indices
        }
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "step":
                rows = []
                for index, action in payload:
                    result = envs[index].step(action)
                    rows.append((index, result, envs[index].action_context()))
                connection.send(("ok", rows))
            elif command == "reset":
                rows = []
                for index, opponent_name in payload:
                    obs = envs[index].reset(opponent_name=opponent_name)
                    rows.append((index, obs, envs[index].action_context()))
                connection.send(("ok", rows))
            elif command == "close":
                connection.send(("ok", None))
                return
            else:
                raise ValueError(f"unknown environment worker command: {command}")
    except (EOFError, KeyboardInterrupt):
        return
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        for env in envs.values():
            env.engine.close()
        connection.close()


class SerialEnvRunner:
    """Existing in-process execution path, kept for debugging and small smokes."""

    backend = "serial"
    num_workers = 1

    def __init__(self, cfg: dict, num_envs: int) -> None:
        first_env = SkyArenaMAPPOEnv(cfg, seed_offset=0)
        shared_pool = first_env.opponent_pool
        self.envs = [first_env] + [
            SkyArenaMAPPOEnv(cfg, seed_offset=index, opponent_pool=shared_pool)
            for index in range(1, num_envs)
        ]
        self.opponent_pool = shared_pool
        self.engine_config = first_env.engine_config
        self.num_agents = first_env.red_fighter_num
        self.obs_shapes = first_env.obs_shapes()
        self.action_contexts: list[SkyArenaActionContext] = []

    def reset_all(self) -> list[dict]:
        observations = [env.reset() for env in self.envs]
        self.action_contexts = [env.action_context() for env in self.envs]
        return observations

    def reset_at(self, index: int) -> dict:
        return self.reset_many([index])[index]

    def reset_many(self, indices: Sequence[int]) -> dict[int, dict]:
        observations = {}
        for index in indices:
            observations[index] = self.envs[index].reset()
            self.action_contexts[index] = self.envs[index].action_context()
        return observations

    def step_all(self, actions: Sequence[Any]) -> list[EnvStep]:
        results = [
            env.step(action)
            for env, action in zip(self.envs, actions, strict=True)
        ]
        self.action_contexts = [env.action_context() for env in self.envs]
        return results

    def close(self) -> None:
        for env in self.envs:
            env.engine.close()


class SubprocessEnvRunner:
    """Persistent worker processes with centralized opponent-pool bookkeeping."""

    backend = "subprocess"

    def __init__(
        self,
        cfg: dict,
        num_envs: int,
        num_workers: int,
        *,
        start_method: str = "forkserver",
    ) -> None:
        if num_workers < 1 or num_workers > num_envs:
            raise ValueError("env_workers must be between 1 and num_envs")
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"unsupported multiprocessing start method: {start_method}")

        reference_env = SkyArenaMAPPOEnv(cfg, seed_offset=0)
        self.opponent_pool = reference_env.opponent_pool
        self.engine_config = reference_env.engine_config
        self.num_agents = reference_env.red_fighter_num
        self.obs_shapes = reference_env.obs_shapes()
        reference_env.engine.close()

        self.num_envs = int(num_envs)
        self.num_workers = int(num_workers)
        self._base_seed = int(cfg["train"].get("seed", 0))
        self.action_contexts: list[SkyArenaActionContext] = []
        self._assignments = [
            list(range(worker_index, self.num_envs, self.num_workers))
            for worker_index in range(self.num_workers)
        ]
        worker_cfg = copy.deepcopy(cfg)
        worker_cfg["env"]["opponent_pool"] = []
        context = mp.get_context(start_method)
        self._connections = []
        self._processes = []
        try:
            for indices in self._assignments:
                parent_connection, child_connection = context.Pipe()
                process = context.Process(
                    target=_worker_main,
                    args=(child_connection, worker_cfg, indices),
                    daemon=True,
                )
                process.start()
                child_connection.close()
                self._connections.append(parent_connection)
                self._processes.append(process)
            for connection in self._connections:
                status, payload = self._receive(connection)
                if status != "ready":
                    self._raise_worker_error(status, payload)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def recommended_workers(num_envs: int) -> int:
        logical_cpus = os.cpu_count() or 2
        return min(int(num_envs), max(1, logical_cpus - 2))

    @staticmethod
    def _receive(connection, timeout: float = 120.0):
        if not connection.poll(timeout):
            raise TimeoutError("environment worker did not respond in time")
        return connection.recv()

    @staticmethod
    def _raise_worker_error(status: str, payload: Any) -> None:
        if status == "error":
            raise RuntimeError(f"environment worker failed:\n{payload}")
        raise RuntimeError(f"unexpected environment worker response: {status!r}")

    def _exchange(self, command: str, payloads: list[list]) -> list:
        for connection, payload in zip(self._connections, payloads, strict=True):
            connection.send((command, payload))
        rows = []
        for connection in self._connections:
            status, payload = self._receive(connection)
            if status != "ok":
                self._raise_worker_error(status, payload)
            if payload:
                rows.extend(payload)
        return rows

    def _opponent_name_for_reset(self) -> str | None:
        if self.opponent_pool is None:
            return None
        return self.opponent_pool.sample_name()

    def _reset_indices(self, indices: Sequence[int]) -> dict[int, dict]:
        requested = list(indices)
        if len(set(requested)) != len(requested):
            raise ValueError("reset indices must be unique")
        opponent_names = {
            index: self._opponent_name_for_reset()
            for index in requested
        }
        payloads = [
            [
                (index, opponent_names[index])
                for index in assignment
                if index in opponent_names
            ]
            for assignment in self._assignments
        ]
        rows = self._exchange("reset", payloads)
        observations: dict[int, dict] = {}
        for index, obs, context in rows:
            observations[index] = obs
            if self.action_contexts:
                self.action_contexts[index] = context
        return observations

    def reset_all(self) -> list[dict]:
        self.action_contexts = [None] * self.num_envs  # type: ignore[list-item]
        observations = self._reset_indices(range(self.num_envs))
        return [observations[index] for index in range(self.num_envs)]

    def reset_at(self, index: int) -> dict:
        return self.reset_many([index])[index]

    def reset_many(self, indices: Sequence[int]) -> dict[int, dict]:
        return self._reset_indices(indices)

    def step_all(self, actions: Sequence[Any]) -> list[EnvStep]:
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {len(actions)}")
        payloads = [
            [(index, actions[index]) for index in assignment]
            for assignment in self._assignments
        ]
        rows = self._exchange("step", payloads)
        results: list[EnvStep | None] = [None] * self.num_envs
        for index, result, context in rows:
            results[index] = result
            self.action_contexts[index] = context
        ordered = [result for result in results if result is not None]
        if len(ordered) != self.num_envs:
            raise RuntimeError("an environment worker returned an incomplete step batch")
        if self.opponent_pool is not None:
            for _obs, _reward, done, info in ordered:
                if done:
                    self.opponent_pool.record_result(
                        str(info.get("opponent_name", "")),
                        str(info.get("winner", "draw")),
                    )
        return ordered

    def close(self) -> None:
        connections = getattr(self, "_connections", [])
        processes = getattr(self, "_processes", [])
        for connection in connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for connection in connections:
            try:
                if connection.poll(2.0):
                    connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
        for process in processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        self._connections = []
        self._processes = []


def build_env_runner(cfg: dict, num_envs: int):
    """Build the configured runner without changing environment semantics."""
    train_cfg = cfg["train"]
    backend = str(train_cfg.get("env_backend", "serial")).lower()
    if backend == "serial":
        return SerialEnvRunner(cfg, num_envs)
    if backend != "subprocess":
        raise ValueError(f"unknown env_backend: {backend}")
    raw_workers = train_cfg.get("env_workers", "auto")
    workers = (
        SubprocessEnvRunner.recommended_workers(num_envs)
        if str(raw_workers).lower() == "auto"
        else int(raw_workers)
    )
    return SubprocessEnvRunner(
        cfg,
        num_envs,
        workers,
        start_method=str(train_cfg.get("env_start_method", "forkserver")),
    )
