from __future__ import annotations

from typing import Any

import numpy as np

from ..core.state import EnvState


class EpisodeSummary:
    """Accumulates per-step data and computes episode-level summary statistics."""

    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []
        self._episode: int | None = None

    def record_step(self, metrics: dict[str, Any], reward_components: dict[str, Any]) -> None:
        """Accumulate step data."""
        self._steps.append({"metrics": dict(metrics), "reward_components": dict(reward_components)})

    def finalize(self, state: EnvState, info: dict[str, Any]) -> dict[str, Any]:
        """Compute episode summary dict from accumulated steps and final state."""
        tracker = state.tracker
        episode_len = len(self._steps)

        # Kills and losses
        red_kills = int(state.red.kills)
        blue_kills = int(state.blue.kills)
        red_losses = int(state.red.losses)
        blue_losses = int(state.blue.losses)

        # Contact and fire opportunity steps
        first_contact_step = tracker.first_contact_step
        first_fire_opportunity_step = tracker.first_fire_opportunity_step
        if first_contact_step is not None and first_fire_opportunity_step is not None:
            contact_to_fire_gap = first_fire_opportunity_step - first_contact_step
        else:
            contact_to_fire_gap = None

        # Expected exchange proxy (mean over steps)
        exchange_values = [
            float(s["metrics"]["expected_exchange_proxy"])
            for s in self._steps
            if s["metrics"].get("expected_exchange_proxy") is not None
        ]
        mean_expected_exchange_proxy = float(np.mean(exchange_values)) if exchange_values else 0.0

        # Selected expected exchange (mean over steps)
        sel_exchange_values = [
            float(s["metrics"]["selected_expected_exchange"])
            for s in self._steps
            if s["metrics"].get("selected_expected_exchange") is not None
        ]
        mean_selected_expected_exchange = float(np.mean(sel_exchange_values)) if sel_exchange_values else 0.0

        # Fire execution rates
        fire_opps_red = int(tracker.fire_opportunities_red)
        fire_exec_red = int(tracker.fire_executions_red)
        mean_red_fire_execution_rate = fire_exec_red / fire_opps_red if fire_opps_red > 0 else 0.0

        fire_opps_blue = int(tracker.fire_opportunities_blue)
        fire_exec_blue = int(tracker.fire_executions_blue)
        mean_blue_fire_execution_rate = fire_exec_blue / fire_opps_blue if fire_opps_blue > 0 else 0.0

        # Invalid fire counts
        red_invalid_fire_count = int(tracker.invalid_fire_count_red)
        blue_invalid_fire_count = int(tracker.invalid_fire_count_blue)

        # Overkill means
        red_overkill = tracker.selected_overkill_values_red
        blue_overkill = tracker.selected_overkill_values_blue
        red_selected_overkill_mean = float(np.mean(red_overkill)) if red_overkill else 0.0
        blue_selected_overkill_mean = float(np.mean(blue_overkill)) if blue_overkill else 0.0

        return {
            "episode": state.episode_idx,
            "winner": state.winner,
            "reason": state.termination_reason,
            "episode_len": episode_len,
            "red_kills": red_kills,
            "blue_kills": blue_kills,
            "red_losses": red_losses,
            "blue_losses": blue_losses,
            "first_contact_step": first_contact_step,
            "first_fire_opportunity_step": first_fire_opportunity_step,
            "contact_to_fire_gap": contact_to_fire_gap,
            "mean_expected_exchange_proxy": mean_expected_exchange_proxy,
            "mean_selected_expected_exchange": mean_selected_expected_exchange,
            "mean_red_fire_execution_rate": mean_red_fire_execution_rate,
            "mean_blue_fire_execution_rate": mean_blue_fire_execution_rate,
            "red_invalid_fire_count": red_invalid_fire_count,
            "blue_invalid_fire_count": blue_invalid_fire_count,
            "red_selected_overkill_mean": red_selected_overkill_mean,
            "blue_selected_overkill_mean": blue_selected_overkill_mean,
        }
