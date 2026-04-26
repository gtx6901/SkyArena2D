from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from ..core.state import EnvState


class TraceRecorder:
    """Records per-step trace data to a JSONL file for detailed episode analysis."""

    def __init__(self, trace_dir: str, episode: int) -> None:
        self.episode = episode
        self.trace_path = Path(trace_dir) / f"episode_{episode:04d}.jsonl"
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.file: TextIO | None = open(self.trace_path, "w", encoding="utf-8")

    def record_step(
        self,
        state: EnvState,
        metrics: dict[str, Any],
        reward_components: dict[str, Any],
    ) -> None:
        """Write one JSON line per step."""
        if self.file is None:
            return

        cache = state.cache
        if cache is None:
            return

        record = {
            "episode": self.episode,
            "step": state.step_count,
            # Alive status
            "red_alive": state.red.alive.tolist(),
            "blue_alive": state.blue.alive.tolist(),
            # Positions
            "red_pos": state.red.pos.tolist(),
            "blue_pos": state.blue.pos.tolist(),
            # Headings
            "red_heading": state.red.heading.tolist(),
            "blue_heading": state.blue.heading.tolist(),
            # Visible matrices
            "red_visible": _to_list(cache.red_visible),
            "blue_visible": _to_list(cache.blue_visible),
            # Ammo
            "red_long_ammo": state.red.long_ammo.tolist(),
            "red_short_ammo": state.red.short_ammo.tolist(),
            "blue_long_ammo": state.blue.long_ammo.tolist(),
            "blue_short_ammo": state.blue.short_ammo.tolist(),
            # Fireable matrices
            "red_fireable_long": _to_list(cache.red_fireable_long),
            "red_fireable_short": _to_list(cache.red_fireable_short),
            "blue_fireable_long": _to_list(cache.blue_fireable_long),
            "blue_fireable_short": _to_list(cache.blue_fireable_short),
            # Attempted/selected fire per agent
            "red_attempted_fire": _to_list(cache.red_attempted_fire),
            "blue_attempted_fire": _to_list(cache.blue_attempted_fire),
            "red_selected_long": _to_list(cache.red_selected_long),
            "red_selected_short": _to_list(cache.red_selected_short),
            "blue_selected_long": _to_list(cache.blue_selected_long),
            "blue_selected_short": _to_list(cache.blue_selected_short),
            "red_selected_target_idx": _to_list(cache.red_selected_target_idx),
            "blue_selected_target_idx": _to_list(cache.blue_selected_target_idx),
            # Attempted/selected allocation matrices
            "red_attempted_long_matrix": _to_list(cache.red_attempted_long_matrix),
            "red_attempted_short_matrix": _to_list(cache.red_attempted_short_matrix),
            "blue_attempted_long_matrix": _to_list(cache.blue_attempted_long_matrix),
            "blue_attempted_short_matrix": _to_list(cache.blue_attempted_short_matrix),
            "red_selected_long_matrix": _to_list(cache.red_selected_long_matrix),
            "red_selected_short_matrix": _to_list(cache.red_selected_short_matrix),
            "blue_selected_long_matrix": _to_list(cache.blue_selected_long_matrix),
            "blue_selected_short_matrix": _to_list(cache.blue_selected_short_matrix),
            # Launch and resolve records
            "launch_records": [_serialize_record(r) for r in cache.launch_records],
            "resolved_records": list(cache.resolved_records),
            # Reward and metrics
            "reward_components": dict(reward_components),
            "metrics": dict(metrics),
            # Terminal state
            "winner": state.winner,
            "reason": state.termination_reason,
        }

        self.file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        """Flush and close the file."""
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None


def _to_list(arr: np.ndarray) -> list:
    """Convert numpy array to nested Python list."""
    if arr is None or (hasattr(arr, "size") and arr.size == 0):
        return []
    return arr.tolist()


def _serialize_record(record: Any) -> Any:
    """Serialize a dataclass record to a dict."""
    if hasattr(record, "__dataclass_fields__"):
        return {k: getattr(record, k) for k in record.__dataclass_fields__}
    return str(record)
