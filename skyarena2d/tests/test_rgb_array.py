from __future__ import annotations

import numpy as np

from skyarena2d.core.config import EnvConfig
from skyarena2d.core.engine import SkyArenaEngine


def test_rgb_array_render_returns_uint8_hwc() -> None:
    cfg = EnvConfig()
    cfg.render.width = 640
    cfg.render.height = 360
    env = SkyArenaEngine(cfg, render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render("rgb_array")
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (360, 640, 3)
    assert frame.dtype == np.uint8
    env.close()
