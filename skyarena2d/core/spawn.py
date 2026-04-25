from __future__ import annotations

import numpy as np

from .config import EnvConfig, SpawnConfig


def _linspace_with_jitter(
    start: float,
    stop: float,
    count: int,
    jitter: float,
    rng: np.random.Generator,
    low: float,
    high: float,
) -> np.ndarray:
    if count == 0:
        return np.zeros((0,), dtype=np.float32)
    base = np.linspace(start, stop, count, dtype=np.float32)
    if jitter > 0:
        base = base + rng.uniform(-jitter, jitter, size=count).astype(np.float32)
    return np.clip(base, low, high)


def _sample_spread_y(
    count: int,
    height: float,
    min_gap: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if count == 0:
        return np.zeros((0,), dtype=np.float32)
    if count == 1:
        return np.array([height * 0.5], dtype=np.float32)

    for _ in range(64):
        ys = np.sort(rng.uniform(0.05 * height, 0.95 * height, size=count)).astype(np.float32)
        diffs = np.diff(ys)
        if np.all(diffs >= min_gap):
            return ys

    ys = np.linspace(0.1 * height, 0.9 * height, count, dtype=np.float32)
    ys = ys + rng.uniform(-0.03 * height, 0.03 * height, size=count).astype(np.float32)
    return np.clip(ys, 0.0, height)


def _spawn_fixed_scaled(
    cfg: SpawnConfig,
    width: float,
    height: float,
    red_count: int,
    blue_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    red_x = np.full((red_count,), width * cfg.red_x_ratio, dtype=np.float32)
    blue_x = np.full((blue_count,), width * cfg.blue_x_ratio, dtype=np.float32)

    y_low = height * cfg.y_min_ratio
    y_high = height * cfg.y_max_ratio
    red_y = _linspace_with_jitter(y_low, y_high, red_count, cfg.jitter, rng, 0.0, height)
    blue_y = _linspace_with_jitter(y_low, y_high, blue_count, cfg.jitter, rng, 0.0, height)

    red_pos = np.stack([red_x, red_y], axis=1) if red_count else np.zeros((0, 2), dtype=np.float32)
    blue_pos = np.stack([blue_x, blue_y], axis=1) if blue_count else np.zeros((0, 2), dtype=np.float32)
    return red_pos, blue_pos


def _spawn_random_edge(
    cfg: SpawnConfig,
    width: float,
    height: float,
    red_count: int,
    blue_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    margin = width * cfg.edge_margin_ratio
    y_gap = max(2.0, height * cfg.min_y_gap_ratio)

    red_x = rng.uniform(0.0, margin, size=red_count).astype(np.float32)
    blue_x = rng.uniform(width - margin, width, size=blue_count).astype(np.float32)
    red_y = _sample_spread_y(red_count, height, y_gap, rng)
    blue_y = _sample_spread_y(blue_count, height, y_gap, rng)

    red_pos = np.stack([red_x, red_y], axis=1) if red_count else np.zeros((0, 2), dtype=np.float32)
    blue_pos = np.stack([blue_x, blue_y], axis=1) if blue_count else np.zeros((0, 2), dtype=np.float32)
    return red_pos, blue_pos


def _spawn_symmetric_random(
    width: float,
    height: float,
    red_count: int,
    blue_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    red_x = rng.uniform(0.08 * width, 0.45 * width, size=red_count).astype(np.float32)
    red_y = rng.uniform(0.08 * height, 0.92 * height, size=red_count).astype(np.float32)
    red_pos = np.stack([red_x, red_y], axis=1) if red_count else np.zeros((0, 2), dtype=np.float32)

    if blue_count == red_count:
        blue_pos = np.empty_like(red_pos)
        blue_pos[:, 0] = width - red_pos[:, 0]
        blue_pos[:, 1] = height - red_pos[:, 1]
    else:
        blue_x = rng.uniform(0.55 * width, 0.92 * width, size=blue_count).astype(np.float32)
        blue_y = rng.uniform(0.08 * height, 0.92 * height, size=blue_count).astype(np.float32)
        blue_pos = (
            np.stack([blue_x, blue_y], axis=1) if blue_count else np.zeros((0, 2), dtype=np.float32)
        )
    return red_pos, blue_pos


def _spawn_curriculum(
    env_cfg: EnvConfig,
    red_count: int,
    blue_count: int,
    rng: np.random.Generator,
    episode_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    stages = sorted(env_cfg.spawn.curriculum, key=lambda s: s.until_episode)
    if not stages:
        fallback = env_cfg.spawn.model_copy(deep=True)
        fallback.mode = "fixed_scaled"
        return _spawn_fixed_scaled(
            fallback,
            env_cfg.map.width,
            env_cfg.map.height,
            red_count,
            blue_count,
            rng,
        )

    stage = stages[-1]
    for candidate in stages:
        if episode_idx <= candidate.until_episode:
            stage = candidate
            break

    merged = env_cfg.spawn.model_copy(deep=True)
    merged.mode = stage.mode
    if stage.red_x_ratio is not None:
        merged.red_x_ratio = stage.red_x_ratio
    if stage.blue_x_ratio is not None:
        merged.blue_x_ratio = stage.blue_x_ratio
    if stage.y_min_ratio is not None:
        merged.y_min_ratio = stage.y_min_ratio
    if stage.y_max_ratio is not None:
        merged.y_max_ratio = stage.y_max_ratio
    if stage.jitter is not None:
        merged.jitter = stage.jitter

    if merged.mode == "fixed_scaled":
        return _spawn_fixed_scaled(
            merged,
            env_cfg.map.width,
            env_cfg.map.height,
            red_count,
            blue_count,
            rng,
        )
    if merged.mode == "random_edge":
        return _spawn_random_edge(
            merged,
            env_cfg.map.width,
            env_cfg.map.height,
            red_count,
            blue_count,
            rng,
        )
    return _spawn_symmetric_random(
        env_cfg.map.width,
        env_cfg.map.height,
        red_count,
        blue_count,
        rng,
    )


def generate_spawn_positions(
    env_cfg: EnvConfig,
    red_count: int,
    blue_count: int,
    rng: np.random.Generator,
    episode_idx: int,
    env_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    _ = env_step
    mode = env_cfg.spawn.mode
    width, height = env_cfg.map.width, env_cfg.map.height

    if mode == "fixed_scaled":
        return _spawn_fixed_scaled(env_cfg.spawn, width, height, red_count, blue_count, rng)
    if mode == "random_edge":
        return _spawn_random_edge(env_cfg.spawn, width, height, red_count, blue_count, rng)
    if mode == "symmetric_random":
        return _spawn_symmetric_random(width, height, red_count, blue_count, rng)
    return _spawn_curriculum(env_cfg, red_count, blue_count, rng, episode_idx)
