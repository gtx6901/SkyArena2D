from __future__ import annotations

import json

from skyarena2d.core.config import load_config
from skyarena2d.core.engine import SkyArenaEngine
from skyarena2d.logging.trace_recorder import TraceRecorder


def test_trace_recorder_includes_visible_and_matrix_fields(tmp_path):
    config = load_config("configs/env_10v10_full.yaml")
    env = SkyArenaEngine(config)
    env.reset(seed=0)
    _, _, _, _, info = env.step({"red": {}, "blue": {}})

    recorder = TraceRecorder(trace_dir=str(tmp_path), episode=0)
    recorder.record_step(env.get_state(), info.get("metrics", {}), info.get("reward_components", {}))
    recorder.close()

    trace_file = tmp_path / "episode_0000.jsonl"
    with trace_file.open("r", encoding="utf-8") as f:
        line = f.readline().strip()

    payload = json.loads(line)
    assert "red_visible" in payload
    assert "blue_visible" in payload
    assert "red_attempted_long_matrix" in payload
    assert "red_attempted_short_matrix" in payload
    assert "blue_attempted_long_matrix" in payload
    assert "blue_attempted_short_matrix" in payload
    assert "red_selected_long_matrix" in payload
    assert "red_selected_short_matrix" in payload
    assert "blue_selected_long_matrix" in payload
    assert "blue_selected_short_matrix" in payload
