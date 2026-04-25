from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.state import TeamState


@dataclass(slots=True)
class DecodedSideAction:
    fighter_action: np.ndarray
    detector_action: np.ndarray
    course: np.ndarray
    radar_freq: np.ndarray
    jammer_freq: np.ndarray
    hit_target: np.ndarray


def _safe_array(action: np.ndarray | list[list[float]] | None, shape: tuple[int, int]) -> np.ndarray:
    if action is None:
        return np.zeros(shape, dtype=np.float32)
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape == shape:
        return arr

    out = np.zeros(shape, dtype=np.float32)
    rows = min(shape[0], arr.shape[0] if arr.ndim > 0 else 0)
    cols = min(shape[1], arr.shape[1] if arr.ndim > 1 else 0)
    if rows > 0 and cols > 0:
        out[:rows, :cols] = arr[:rows, :cols]
    return out


def decode_maca_side_action(
    team: TeamState,
    fighter_action: np.ndarray | list[list[float]] | None,
    detector_action: np.ndarray | list[list[float]] | None,
    freq_count: int,
) -> DecodedSideAction:
    fighter_arr = _safe_array(fighter_action, (team.num_fighters, 4))
    detector_arr = _safe_array(detector_action, (team.num_detectors, 2))

    course = np.zeros((team.total_units,), dtype=np.float32)
    radar_freq = np.zeros((team.total_units,), dtype=np.int32)
    jammer_freq = np.zeros((team.total_units,), dtype=np.int32)
    hit_target = np.zeros((team.total_units,), dtype=np.int32)

    if team.num_fighters > 0:
        course[: team.num_fighters] = fighter_arr[:, 0] % 360.0
        radar_freq[: team.num_fighters] = np.clip(
            fighter_arr[:, 1].astype(np.int32), 0, freq_count
        )
        jammer_freq[: team.num_fighters] = np.clip(
            fighter_arr[:, 2].astype(np.int32), 0, freq_count + 1
        )
        hit_target[: team.num_fighters] = np.maximum(fighter_arr[:, 3].astype(np.int32), 0)

    if team.num_detectors > 0:
        start = team.num_fighters
        stop = team.total_units
        course[start:stop] = detector_arr[:, 0] % 360.0
        radar_freq[start:stop] = np.clip(detector_arr[:, 1].astype(np.int32), 0, freq_count)
        jammer_freq[start:stop] = 0
        hit_target[start:stop] = 0

    return DecodedSideAction(
        fighter_action=fighter_arr,
        detector_action=detector_arr,
        course=course,
        radar_freq=radar_freq,
        jammer_freq=jammer_freq,
        hit_target=hit_target,
    )
