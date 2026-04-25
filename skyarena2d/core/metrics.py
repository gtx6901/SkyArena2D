from __future__ import annotations

from collections import defaultdict

import numpy as np

from .state import EnvState
from .weapons import WeaponStepResult


def _expected_kill_proxy(
    fireable_long: np.ndarray,
    fireable_short: np.ndarray,
    long_p: np.ndarray,
    short_p: np.ndarray,
) -> tuple[float, int, float, int, int]:
    if fireable_long.size == 0:
        return 0.0, 0, 0.0, 0, 0

    fireable_any = fireable_long | fireable_short
    edges = int(np.count_nonzero(fireable_any))
    fireable_agents = int(np.count_nonzero(np.any(fireable_any, axis=1)))
    target_n = fireable_any.shape[1]

    expected = 0.0
    unique_targets = 0
    overkill_vals: list[float] = []

    for target_idx in range(target_n):
        incoming_long = np.where(fireable_long[:, target_idx])[0]
        incoming_short = np.where(fireable_short[:, target_idx])[0]
        if incoming_long.size == 0 and incoming_short.size == 0:
            continue

        probs: list[float] = []
        if incoming_long.size:
            probs.extend(long_p[incoming_long].tolist())
        if incoming_short.size:
            probs.extend(short_p[incoming_short].tolist())
        probs_arr = np.clip(np.array(probs, dtype=np.float32), 0.0, 1.0)
        kill_prob = 1.0 - float(np.prod(1.0 - probs_arr))
        expected += kill_prob
        unique_targets += 1
        overkill_vals.append(max(0.0, float(len(probs) - 1)))

    overkill_mean = float(np.mean(overkill_vals)) if overkill_vals else 0.0
    return expected, unique_targets, overkill_mean, edges, fireable_agents


def _selected_expected_kills(
    selected_long: np.ndarray,
    selected_short: np.ndarray,
    selected_target_idx: np.ndarray,
    long_p: np.ndarray,
    short_p: np.ndarray,
    num_enemies: int,
) -> tuple[float, int, float]:
    """Compute expected kills based on actually selected (launched) targets."""
    if selected_long.size == 0 or num_enemies == 0:
        return 0.0, 0, 0.0

    # group by target
    target_probs: dict[int, list[float]] = defaultdict(list)
    for i in range(len(selected_long)):
        t = int(selected_target_idx[i])
        if t < 0:
            continue
        if bool(selected_long[i]) and i < len(long_p):
            target_probs[t].append(float(long_p[i]))
        elif bool(selected_short[i]) and i < len(short_p):
            target_probs[t].append(float(short_p[i]))

    expected = 0.0
    unique_targets = len(target_probs)
    overkill_vals: list[float] = []
    for t, probs in target_probs.items():
        probs_arr = np.clip(np.array(probs, dtype=np.float32), 0.0, 1.0)
        kill_prob = 1.0 - float(np.prod(1.0 - probs_arr))
        expected += kill_prob
        overkill_vals.append(max(0.0, float(len(probs) - 1)))

    overkill_mean = float(np.mean(overkill_vals)) if overkill_vals else 0.0
    return expected, unique_targets, overkill_mean


def update_tracker(
    *,
    state: EnvState,
    weapon_result: WeaponStepResult,
    red_visible: np.ndarray,
    blue_visible: np.ndarray,
    red_jammed_count: int,
    blue_jammed_count: int,
    passive_count_red: int,
    passive_count_blue: int,
) -> None:
    tracker = state.tracker

    if tracker.first_contact_step is None and (np.any(red_visible) or np.any(blue_visible)):
        tracker.first_contact_step = state.step_count

    red_has_fire = np.any(weapon_result.red_fireable_long | weapon_result.red_fireable_short)
    blue_has_fire = np.any(weapon_result.blue_fireable_long | weapon_result.blue_fireable_short)
    if tracker.first_fire_opportunity_step is None and (red_has_fire or blue_has_fire):
        tracker.first_fire_opportunity_step = state.step_count

    tracker.missiles_launched_long += weapon_result.missiles_launched_long
    tracker.missiles_launched_short += weapon_result.missiles_launched_short
    tracker.missiles_hit += weapon_result.missiles_hit
    tracker.missiles_missed += weapon_result.missiles_missed
    tracker.jammed_detection_count += red_jammed_count + blue_jammed_count
    tracker.passive_detection_count += passive_count_red + passive_count_blue

    incoming_count_red = defaultdict(int)
    incoming_count_blue = defaultdict(int)

    for launch in weapon_result.launch_records:
        if not launch.valid:
            continue
        if launch.attacker_side == "red":
            tracker.unique_targets_engaged_red.add(launch.target_idx)
            incoming_count_blue[launch.target_idx] += 1
        else:
            tracker.unique_targets_engaged_blue.add(launch.target_idx)
            incoming_count_red[launch.target_idx] += 1

    for count in incoming_count_blue.values():
        tracker.overkill_values_red.append(max(0, count - 1))
    for count in incoming_count_red.values():
        tracker.overkill_values_blue.append(max(0, count - 1))

    # --- new: attempted/selected tracking ---
    if weapon_result.red_attempted_fire.size > 0:
        tracker.fire_attempts_red += int(np.count_nonzero(weapon_result.red_attempted_fire))
    if weapon_result.blue_attempted_fire.size > 0:
        tracker.fire_attempts_blue += int(np.count_nonzero(weapon_result.blue_attempted_fire))

    # fire opportunities = agents with any fireable edge
    if weapon_result.red_fireable_long.size > 0:
        red_opp = int(np.count_nonzero(np.any(
            weapon_result.red_fireable_long | weapon_result.red_fireable_short, axis=1
        )))
        tracker.fire_opportunities_red += red_opp
    if weapon_result.blue_fireable_long.size > 0:
        blue_opp = int(np.count_nonzero(np.any(
            weapon_result.blue_fireable_long | weapon_result.blue_fireable_short, axis=1
        )))
        tracker.fire_opportunities_blue += blue_opp

    # fire executions = agents that actually fired (valid)
    if weapon_result.red_valid_fire.size > 0:
        tracker.fire_executions_red += int(np.count_nonzero(weapon_result.red_valid_fire))
    if weapon_result.blue_valid_fire.size > 0:
        tracker.fire_executions_blue += int(np.count_nonzero(weapon_result.blue_valid_fire))

    # invalid fire counts
    if weapon_result.red_invalid_fire.size > 0:
        tracker.invalid_fire_count_red += int(np.count_nonzero(weapon_result.red_invalid_fire))
    if weapon_result.blue_invalid_fire.size > 0:
        tracker.invalid_fire_count_blue += int(np.count_nonzero(weapon_result.blue_invalid_fire))

    # selected targets tracking
    if weapon_result.red_selected_target_idx.size > 0:
        for t in weapon_result.red_selected_target_idx:
            if int(t) >= 0:
                tracker.unique_selected_targets_red.add(int(t))
    if weapon_result.blue_selected_target_idx.size > 0:
        for t in weapon_result.blue_selected_target_idx:
            if int(t) >= 0:
                tracker.unique_selected_targets_blue.add(int(t))

    # selected overkill
    sel_incoming_blue: dict[int, int] = defaultdict(int)
    sel_incoming_red: dict[int, int] = defaultdict(int)
    if weapon_result.red_selected_target_idx.size > 0:
        for t in weapon_result.red_selected_target_idx:
            if int(t) >= 0:
                sel_incoming_blue[int(t)] += 1
    if weapon_result.blue_selected_target_idx.size > 0:
        for t in weapon_result.blue_selected_target_idx:
            if int(t) >= 0:
                sel_incoming_red[int(t)] += 1
    for count in sel_incoming_blue.values():
        tracker.selected_overkill_values_red.append(max(0, count - 1))
    for count in sel_incoming_red.values():
        tracker.selected_overkill_values_blue.append(max(0, count - 1))


def build_metrics_snapshot(state: EnvState, weapon_result: WeaponStepResult) -> dict[str, float | int | None]:
    red_expected, red_unique, red_overkill, red_edges, red_fireable_agents = _expected_kill_proxy(
        weapon_result.red_fireable_long,
        weapon_result.red_fireable_short,
        state.red.long_hit_prob,
        state.red.short_hit_prob,
    )
    blue_expected, blue_unique, blue_overkill, blue_edges, blue_fireable_agents = _expected_kill_proxy(
        weapon_result.blue_fireable_long,
        weapon_result.blue_fireable_short,
        state.blue.long_hit_prob,
        state.blue.short_hit_prob,
    )

    # selected expected kills
    num_blue = state.blue.num_fighters
    num_red = state.red.num_fighters
    red_sel_exp, red_sel_unique, red_sel_overkill = _selected_expected_kills(
        weapon_result.red_selected_long,
        weapon_result.red_selected_short,
        weapon_result.red_selected_target_idx,
        state.red.long_hit_prob,
        state.red.short_hit_prob,
        num_blue,
    )
    blue_sel_exp, blue_sel_unique, blue_sel_overkill = _selected_expected_kills(
        weapon_result.blue_selected_long,
        weapon_result.blue_selected_short,
        weapon_result.blue_selected_target_idx,
        state.blue.long_hit_prob,
        state.blue.short_hit_prob,
        num_red,
    )

    # attempted/selected edges
    red_attempted_edges = int(np.count_nonzero(weapon_result.red_attempted_fire)) if weapon_result.red_attempted_fire.size > 0 else 0
    blue_attempted_edges = int(np.count_nonzero(weapon_result.blue_attempted_fire)) if weapon_result.blue_attempted_fire.size > 0 else 0
    red_selected_edges = int(np.count_nonzero(weapon_result.red_selected_long | weapon_result.red_selected_short)) if weapon_result.red_selected_long.size > 0 else 0
    blue_selected_edges = int(np.count_nonzero(weapon_result.blue_selected_long | weapon_result.blue_selected_short)) if weapon_result.blue_selected_long.size > 0 else 0

    tracker = state.tracker
    contact_gap = None
    if tracker.first_contact_step is not None and tracker.first_fire_opportunity_step is not None:
        contact_gap = tracker.first_fire_opportunity_step - tracker.first_contact_step

    # fire execution rate given opportunity
    red_exec_rate = (
        float(tracker.fire_executions_red) / float(tracker.fire_opportunities_red)
        if tracker.fire_opportunities_red > 0 else 0.0
    )
    blue_exec_rate = (
        float(tracker.fire_executions_blue) / float(tracker.fire_opportunities_blue)
        if tracker.fire_opportunities_blue > 0 else 0.0
    )

    return {
        # --- original fields ---
        "red_alive": state.red.alive_count,
        "blue_alive": state.blue.alive_count,
        "red_kills": state.red.kills,
        "blue_kills": state.blue.kills,
        "red_losses": state.red.losses,
        "blue_losses": state.blue.losses,
        "red_fireable_edges": red_edges,
        "blue_fireable_edges": blue_edges,
        "fireability_edge_advantage": red_edges - blue_edges,
        "red_fireable_agents": red_fireable_agents,
        "blue_fireable_agents": blue_fireable_agents,
        "expected_red_kills_proxy": red_expected,
        "expected_blue_kills_proxy": blue_expected,
        "expected_exchange_proxy": red_expected - blue_expected,
        "unique_targets_engaged_red": len(tracker.unique_targets_engaged_red) or red_unique,
        "unique_targets_engaged_blue": len(tracker.unique_targets_engaged_blue) or blue_unique,
        "overkill_mean_red": float(np.mean(tracker.overkill_values_red))
        if tracker.overkill_values_red
        else red_overkill,
        "overkill_mean_blue": float(np.mean(tracker.overkill_values_blue))
        if tracker.overkill_values_blue
        else blue_overkill,
        "first_contact_step": tracker.first_contact_step,
        "first_fire_opportunity_step": tracker.first_fire_opportunity_step,
        "contact_to_fire_gap": contact_gap,
        "missiles_launched_long": tracker.missiles_launched_long,
        "missiles_launched_short": tracker.missiles_launched_short,
        "missiles_hit": tracker.missiles_hit,
        "missiles_missed": tracker.missiles_missed,
        "jammed_detection_count": tracker.jammed_detection_count,
        "passive_detection_count": tracker.passive_detection_count,
        # --- new: attempted/selected ---
        "red_attempted_edges": red_attempted_edges,
        "blue_attempted_edges": blue_attempted_edges,
        "red_selected_edges": red_selected_edges,
        "blue_selected_edges": blue_selected_edges,
        "selected_edge_advantage": red_selected_edges - blue_selected_edges,
        "red_fire_execution_rate_given_opportunity": red_exec_rate,
        "blue_fire_execution_rate_given_opportunity": blue_exec_rate,
        "selected_expected_red_kills": red_sel_exp,
        "selected_expected_blue_kills": blue_sel_exp,
        "selected_expected_exchange": red_sel_exp - blue_sel_exp,
        "red_unique_selected_targets": len(tracker.unique_selected_targets_red) or red_sel_unique,
        "blue_unique_selected_targets": len(tracker.unique_selected_targets_blue) or blue_sel_unique,
        "red_selected_overkill_mean": float(np.mean(tracker.selected_overkill_values_red))
        if tracker.selected_overkill_values_red else red_sel_overkill,
        "blue_selected_overkill_mean": float(np.mean(tracker.selected_overkill_values_blue))
        if tracker.selected_overkill_values_blue else blue_sel_overkill,
        "red_invalid_fire_count": tracker.invalid_fire_count_red,
        "blue_invalid_fire_count": tracker.invalid_fire_count_blue,
        "red_fire_attempts": tracker.fire_attempts_red,
        "blue_fire_attempts": tracker.fire_attempts_blue,
    }
