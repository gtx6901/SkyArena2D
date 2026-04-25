from __future__ import annotations

import numpy as np


def _expected_from_edges(
    long_edges: np.ndarray,
    short_edges: np.ndarray,
    long_prob: np.ndarray,
    short_prob: np.ndarray,
) -> float:
    if long_edges.size == 0:
        return 0.0
    target_n = long_edges.shape[1]
    total = 0.0
    for t in range(target_n):
        p: list[float] = []
        i_long = np.where(long_edges[:, t])[0]
        i_short = np.where(short_edges[:, t])[0]
        if i_long.size:
            p.extend(long_prob[i_long].tolist())
        if i_short.size:
            p.extend(short_prob[i_short].tolist())
        if not p:
            continue
        arr = np.clip(np.array(p, dtype=np.float32), 0.0, 1.0)
        total += 1.0 - float(np.prod(1.0 - arr))
    return float(total)


def fireability_summary(
    red_long: np.ndarray,
    red_short: np.ndarray,
    blue_long: np.ndarray,
    blue_short: np.ndarray,
    red_long_prob: np.ndarray,
    red_short_prob: np.ndarray,
    blue_long_prob: np.ndarray,
    blue_short_prob: np.ndarray,
) -> dict[str, float | int]:
    red_edges = int(np.count_nonzero(red_long | red_short))
    blue_edges = int(np.count_nonzero(blue_long | blue_short))
    red_agents = int(np.count_nonzero(np.any(red_long | red_short, axis=1))) if red_long.size else 0
    blue_agents = int(np.count_nonzero(np.any(blue_long | blue_short, axis=1))) if blue_long.size else 0

    red_expected = _expected_from_edges(red_long, red_short, red_long_prob, red_short_prob)
    blue_expected = _expected_from_edges(blue_long, blue_short, blue_long_prob, blue_short_prob)

    return {
        "red_fireable_edges": red_edges,
        "blue_fireable_edges": blue_edges,
        "fireability_edge_advantage": red_edges - blue_edges,
        "red_fireable_agents": red_agents,
        "blue_fireable_agents": blue_agents,
        "expected_red_kills_proxy": red_expected,
        "expected_blue_kills_proxy": blue_expected,
        "expected_exchange_proxy": red_expected - blue_expected,
    }
