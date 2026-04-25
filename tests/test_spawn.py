from __future__ import annotations

import numpy as np

from skyarena2d.core.config import EnvConfig
from skyarena2d.core.spawn import generate_spawn_positions


def test_fixed_scaled_y_scales_with_map_height() -> None:
    cfg_small = EnvConfig()
    cfg_small.map.height = 1000
    cfg_small.map.width = 2000
    cfg_small.spawn.mode = "fixed_scaled"
    cfg_small.spawn.jitter = 0.0
    rng = np.random.default_rng(0)

    _, blue_small = generate_spawn_positions(cfg_small, 10, 10, rng, 0, 0)

    cfg_big = cfg_small.model_copy(deep=True)
    cfg_big.map.height = 2000
    rng2 = np.random.default_rng(0)
    _, blue_big = generate_spawn_positions(cfg_big, 10, 10, rng2, 0, 0)

    assert np.allclose(blue_big[:, 1], blue_small[:, 1] * 2.0)


def test_symmetric_random_is_center_mirrored() -> None:
    cfg = EnvConfig()
    cfg.map.width = 1600
    cfg.map.height = 900
    cfg.spawn.mode = "symmetric_random"
    rng = np.random.default_rng(42)

    red, blue = generate_spawn_positions(cfg, 8, 8, rng, 0, 0)

    assert np.allclose(blue[:, 0], cfg.map.width - red[:, 0])
    assert np.allclose(blue[:, 1], cfg.map.height - red[:, 1])


def test_random_edge_within_bounds() -> None:
    cfg = EnvConfig()
    cfg.spawn.mode = "random_edge"
    cfg.map.width = 1800
    cfg.map.height = 1000
    rng = np.random.default_rng(7)

    red, blue = generate_spawn_positions(cfg, 10, 10, rng, 0, 0)

    assert np.all((red[:, 0] >= 0) & (red[:, 0] <= cfg.map.width))
    assert np.all((blue[:, 0] >= 0) & (blue[:, 0] <= cfg.map.width))
    assert np.all((red[:, 1] >= 0) & (red[:, 1] <= cfg.map.height))
    assert np.all((blue[:, 1] >= 0) & (blue[:, 1] <= cfg.map.height))
