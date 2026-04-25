from __future__ import annotations

import numpy as np
import pytest

from skyarena2d.core.config import load_config
from skyarena2d.core.engine import SkyArenaEngine
from skyarena2d.render.pixel_renderer import PixelRenderer


def test_fallback_rgb_shape_unchanged():
    config = load_config("configs/env_10v10_full.yaml")
    env = SkyArenaEngine(config, render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    env.step({"red": {}, "blue": {}})

    renderer = PixelRenderer(config)
    state = env.get_state()
    frame = renderer._fallback_rgb(state)

    assert frame.shape == (renderer.render_height, renderer.render_width, 3)
    assert frame.dtype == np.uint8


def test_fallback_rgb_scoreboard_area_dark():
    config = load_config("configs/env_10v10_full.yaml")
    env = SkyArenaEngine(config, render_mode="rgb_array")
    obs, info = env.reset(seed=42)

    renderer = PixelRenderer(config)
    state = env.get_state()
    frame = renderer._fallback_rgb(state)

    # Top scoreboard_height rows should be darker (scoreboard background)
    scoreboard_area = frame[: renderer.scoreboard_height, :, :]
    map_area = frame[renderer.scoreboard_height :, :, :]

    # Verify scoreboard area is not all zeros
    assert np.any(scoreboard_area > 0)

    # Verify map area exists
    assert map_area.shape[0] == renderer.render_height - renderer.scoreboard_height


def test_world_to_screen_avoids_scoreboard():
    config = load_config("configs/env_10v10_full.yaml")
    renderer = PixelRenderer(config)

    # World (0, 0) should map to y >= scoreboard_height
    sx, sy = renderer._world_to_screen(0, 0)
    assert sy >= renderer.scoreboard_height

    # World (0, map_height) should map to y <= render_height
    sx, sy = renderer._world_to_screen(0, config.map.height)
    assert sy <= renderer.render_height

    # All y coordinates should be in valid range
    for y_world in [0, config.map.height / 2, config.map.height]:
        sx, sy = renderer._world_to_screen(0, y_world)
        assert renderer.scoreboard_height <= sy <= renderer.render_height


def test_render_rgb_array_shape():
    config = load_config("configs/env_10v10_full.yaml")
    env = SkyArenaEngine(config, render_mode="rgb_array")
    obs, info = env.reset(seed=42)

    frame = env.render()

    # Should return (H, W, 3) where H == render_height (scoreboard included)
    assert frame is not None
    assert frame.shape[0] == config.render.height
    assert frame.shape[1] == config.render.width
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
