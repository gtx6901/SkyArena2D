from __future__ import annotations

import pytest

from skyarena2d.rl.algo.mappo_trainer import _accumulate_eval_step_metrics


def test_eval_step_metrics_are_summed_across_episode() -> None:
    totals: dict[str, float] = {}

    _accumulate_eval_step_metrics(
        totals,
        {
            "red_fireable_edges": 3,
            "red_attempted_edges": 2,
            "red_selected_edges": 1,
            "selected_expected_exchange": 0.4,
            "red_invalid_fire_count": 7,
        },
    )
    _accumulate_eval_step_metrics(
        totals,
        {
            "red_fireable_edges": 5,
            "red_attempted_edges": 1,
            "red_selected_edges": 1,
            "selected_expected_exchange": -0.1,
            "red_invalid_fire_count": 9,
        },
    )

    assert totals["red_fireable_edges"] == 8
    assert totals["red_attempted_edges"] == 3
    assert totals["red_selected_edges"] == 2
    assert totals["selected_expected_exchange"] == pytest.approx(0.3)
    assert "red_invalid_fire_count" not in totals
