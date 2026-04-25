from __future__ import annotations

from skyarena2d.core.config import EnvConfig
from skyarena2d.core.engine import SkyArenaEngine


def test_modern_obs_shapes_and_masks() -> None:
    cfg = EnvConfig()
    cfg.teams.red_fighters = 3
    cfg.teams.blue_fighters = 4
    cfg.teams.red_detectors = 1
    cfg.teams.blue_detectors = 0
    cfg.spawn.jitter = 0.0

    env = SkyArenaEngine(cfg)
    obs, _ = env.reset(seed=0)
    red = obs["red"]["modern"]

    assert red["self"].shape == (4, 14)
    assert red["allies"].shape[0] == 4
    assert red["allies"].shape[1] == 3
    assert red["enemies"].shape[0] == 4
    assert red["enemies"].shape[1] == 4
    assert red["masks"]["ally"].shape == (4, 3)
    assert red["masks"]["enemy"].shape == (4, 4)
    assert red["masks"]["self_alive"].shape == (4,)
