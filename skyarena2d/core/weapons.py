from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EnvConfig
from .state import EnvState, LaunchRecord, MissileEvent, TeamState


@dataclass(slots=True)
class WeaponStepResult:
    red_fireable_long: np.ndarray
    red_fireable_short: np.ndarray
    blue_fireable_long: np.ndarray
    blue_fireable_short: np.ndarray
    launch_records: list[LaunchRecord]
    resolved_records: list[dict[str, object]]
    red_valid_fire: np.ndarray
    red_invalid_fire: np.ndarray
    blue_valid_fire: np.ndarray
    blue_invalid_fire: np.ndarray
    killed_red: list[int]
    killed_blue: list[int]
    missiles_launched_long: int
    missiles_launched_short: int
    missiles_hit: int
    missiles_missed: int
    # --- new: attempted = agent submitted fire_action > 0 ---
    red_attempted_fire: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    blue_attempted_fire: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # --- new: selected = attempted AND fireable (actually launched) ---
    red_selected_long: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    red_selected_short: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    blue_selected_long: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    blue_selected_short: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    red_selected_target_idx: np.ndarray = field(default_factory=lambda: np.full(0, -1, dtype=np.int32))
    blue_selected_target_idx: np.ndarray = field(default_factory=lambda: np.full(0, -1, dtype=np.int32))
    red_attempted_long_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    red_attempted_short_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    blue_attempted_long_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    blue_attempted_short_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    red_selected_long_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    red_selected_short_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    blue_selected_long_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    blue_selected_short_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))


def _event_dict(event: MissileEvent) -> dict[str, object]:
    return {
        "attacker_side": event.attacker_side,
        "attacker_idx": event.attacker_idx,
        "target_side": event.target_side,
        "target_idx": event.target_idx,
        "missile_type": event.missile_type,
        "launch_step": event.launch_step,
        "resolve_step": event.resolve_step,
        "hit_prob": event.hit_prob,
    }


def decode_hit_target(hit_target: int, max_enemy: int) -> tuple[str | None, int | None]:
    if hit_target <= 0:
        return None, None
    if 1 <= hit_target <= max_enemy:
        return "long", hit_target - 1
    if max_enemy < hit_target <= 2 * max_enemy:
        return "short", hit_target - max_enemy - 1
    return None, None


def _build_passive_matrix(passive_lists: list[list[dict[str, object]]], enemy_n: int) -> np.ndarray:
    matrix = np.zeros((len(passive_lists), enemy_n), dtype=bool)
    for i, row in enumerate(passive_lists):
        for item in row:
            idx = int(item["id"]) - 1
            if 0 <= idx < enemy_n:
                matrix[i, idx] = True
    return matrix


def compute_fireable_matrix(
    own: TeamState,
    enemy: TeamState,
    visible_matrix: np.ndarray,
    allow_passive_fire: bool,
    passive_lists: list[list[dict[str, object]]],
) -> tuple[np.ndarray, np.ndarray]:
    if own.total_units == 0 or enemy.total_units == 0:
        shape = (own.total_units, enemy.total_units)
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)

    dx = enemy.pos[None, :, 0] - own.pos[:, None, 0]
    dy = enemy.pos[None, :, 1] - own.pos[:, None, 1]
    dist = np.sqrt(dx * dx + dy * dy)

    base = own.alive[:, None] & enemy.alive[None, :]
    base &= own.unit_type[:, None] == 0

    if allow_passive_fire:
        passive_matrix = _build_passive_matrix(passive_lists, enemy.total_units)
        target_known = visible_matrix | passive_matrix
    else:
        target_known = visible_matrix

    long_fireable = (
        base
        & target_known
        & (dist <= own.long_range[:, None])
        & (own.long_ammo[:, None] > 0)
    )
    short_fireable = (
        base
        & target_known
        & (dist <= own.short_range[:, None])
        & (own.short_ammo[:, None] > 0)
    )
    return long_fireable, short_fireable


def _queue_and_collect_launches(
    *,
    state: EnvState,
    config: EnvConfig,
    side_name: str,
    own: TeamState,
    enemy: TeamState,
    hit_targets: np.ndarray,
    fireable_long: np.ndarray,
    fireable_short: np.ndarray,
) -> tuple[list[LaunchRecord], np.ndarray, np.ndarray, list[MissileEvent], int, int,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns launch_records, valid, invalid, events, launched_long, launched_short,
    attempted, selected_long, selected_short, selected_target_idx,
    attempted_long_matrix, attempted_short_matrix, selected_long_matrix, selected_short_matrix."""
    launch_records: list[LaunchRecord] = []
    events: list[MissileEvent] = []
    valid = np.zeros((own.total_units,), dtype=bool)
    invalid = np.zeros((own.total_units,), dtype=bool)
    attempted = np.zeros((own.total_units,), dtype=bool)
    selected_long = np.zeros((own.total_units,), dtype=bool)
    selected_short = np.zeros((own.total_units,), dtype=bool)
    selected_target_idx = np.full((own.total_units,), -1, dtype=np.int32)
    attempted_long_matrix = np.zeros((own.total_units, enemy.total_units), dtype=bool)
    attempted_short_matrix = np.zeros((own.total_units, enemy.total_units), dtype=bool)
    selected_long_matrix = np.zeros((own.total_units, enemy.total_units), dtype=bool)
    selected_short_matrix = np.zeros((own.total_units, enemy.total_units), dtype=bool)
    launched_long = 0
    launched_short = 0

    max_enemy = enemy.total_units
    for i in range(own.num_fighters):
        hit_code = int(hit_targets[i]) if i < len(hit_targets) else 0
        missile_type, target_idx = decode_hit_target(hit_code, max_enemy)
        if missile_type is None or target_idx is None:
            continue

        # agent attempted to fire
        attempted[i] = True
        if missile_type == "long":
            attempted_long_matrix[i, target_idx] = True
        else:
            attempted_short_matrix[i, target_idx] = True

        can_fire = False
        reason = ""
        if missile_type == "long":
            can_fire = bool(fireable_long[i, target_idx])
            if not can_fire:
                reason = "not_fireable_long"
        else:
            can_fire = bool(fireable_short[i, target_idx])
            if not can_fire:
                reason = "not_fireable_short"

        if not can_fire:
            invalid[i] = True
            launch_records.append(
                LaunchRecord(
                    attacker_side=state.red.side if side_name == "red" else state.blue.side,
                    attacker_idx=i,
                    target_side=enemy.side,
                    target_idx=target_idx,
                    missile_type=missile_type,
                    hit_prob=0.0,
                    valid=False,
                    reason=reason,
                )
            )
            continue

        if missile_type == "long":
            own.long_ammo[i] -= 1
            hit_prob = float(own.long_hit_prob[i])
            launched_long += 1
            selected_long[i] = True
            selected_long_matrix[i, target_idx] = True
        else:
            own.short_ammo[i] -= 1
            hit_prob = float(own.short_hit_prob[i])
            launched_short += 1
            selected_short[i] = True
            selected_short_matrix[i, target_idx] = True

        valid[i] = True
        selected_target_idx[i] = target_idx
        launch_records.append(
            LaunchRecord(
                attacker_side=state.red.side if side_name == "red" else state.blue.side,
                attacker_idx=i,
                target_side=enemy.side,
                target_idx=target_idx,
                missile_type=missile_type,
                hit_prob=hit_prob,
                valid=True,
                reason="ok",
            )
        )
        events.append(
            MissileEvent(
                attacker_side=state.red.side if side_name == "red" else state.blue.side,
                attacker_idx=i,
                target_side=enemy.side,
                target_idx=target_idx,
                missile_type=missile_type,
                launch_step=state.step_count,
                resolve_step=state.step_count + config.weapon.attack_effect_delay,
                hit_prob=hit_prob,
            )
        )

    return (launch_records, valid, invalid, events, launched_long, launched_short,
            attempted, selected_long, selected_short, selected_target_idx,
            attempted_long_matrix, attempted_short_matrix, selected_long_matrix, selected_short_matrix)


def _resolve_due_events(
    *,
    state: EnvState,
    due_events: list[MissileEvent],
    hit_prob_enable: bool,
) -> tuple[list[dict[str, object]], list[int], list[int], int, int]:
    if not due_events:
        return [], [], [], 0, 0

    grouped: dict[tuple[str, int], list[MissileEvent]] = {}
    for event in due_events:
        grouped.setdefault((event.target_side, event.target_idx), []).append(event)

    resolved_records: list[dict[str, object]] = []
    killed_red: list[int] = []
    killed_blue: list[int] = []
    missiles_hit = 0
    missiles_missed = 0

    red_alive_snapshot = state.red.alive.copy()
    blue_alive_snapshot = state.blue.alive.copy()

    for (target_side, target_idx), events in grouped.items():
        alive_snapshot = red_alive_snapshot if target_side == "red" else blue_alive_snapshot
        target_alive = bool(alive_snapshot[target_idx]) if target_idx < len(alive_snapshot) else False

        if not target_alive:
            missiles_missed += len(events)
            resolved_records.append(
                {
                    "target_side": target_side,
                    "target_idx": target_idx,
                    "target_destroyed": False,
                    "incoming": [_event_dict(e) for e in events],
                    "reason": "already_dead",
                }
            )
            continue

        probs = np.array(
            [1.0 if not hit_prob_enable else float(np.clip(e.hit_prob, 0.0, 1.0)) for e in events],
            dtype=np.float32,
        )
        kill_prob = 1.0 - float(np.prod(1.0 - probs))
        killed = bool(state.rng.random() < kill_prob)

        if killed:
            missiles_hit += 1
            missiles_missed += max(0, len(events) - 1)
            if target_side == "red":
                killed_red.append(target_idx)
            else:
                killed_blue.append(target_idx)
        else:
            missiles_missed += len(events)

        resolved_records.append(
            {
                "target_side": target_side,
                "target_idx": target_idx,
                "target_destroyed": killed,
                "kill_prob": kill_prob,
                "incoming": [_event_dict(e) for e in events],
                "reason": "resolved",
            }
        )

    if killed_red:
        state.red.alive[np.array(killed_red, dtype=np.int64)] = False
    if killed_blue:
        state.blue.alive[np.array(killed_blue, dtype=np.int64)] = False

    return resolved_records, killed_red, killed_blue, missiles_hit, missiles_missed


def process_weapons(
    *,
    state: EnvState,
    config: EnvConfig,
    red_hit_targets: np.ndarray,
    blue_hit_targets: np.ndarray,
    red_visible: np.ndarray,
    blue_visible: np.ndarray,
    red_passive: list[list[dict[str, object]]],
    blue_passive: list[list[dict[str, object]]],
) -> WeaponStepResult:
    red_fireable_long, red_fireable_short = compute_fireable_matrix(
        state.red,
        state.blue,
        red_visible,
        config.weapon.allow_passive_fire,
        red_passive,
    )
    blue_fireable_long, blue_fireable_short = compute_fireable_matrix(
        state.blue,
        state.red,
        blue_visible,
        config.weapon.allow_passive_fire,
        blue_passive,
    )

    (
        red_launch_records,
        red_valid,
        red_invalid,
        red_events,
        red_launched_long,
        red_launched_short,
        red_attempted,
        red_sel_long,
        red_sel_short,
        red_sel_target,
        red_attempted_long_matrix,
        red_attempted_short_matrix,
        red_selected_long_matrix,
        red_selected_short_matrix,
    ) = _queue_and_collect_launches(
        state=state,
        config=config,
        side_name="red",
        own=state.red,
        enemy=state.blue,
        hit_targets=red_hit_targets,
        fireable_long=red_fireable_long,
        fireable_short=red_fireable_short,
    )
    (
        blue_launch_records,
        blue_valid,
        blue_invalid,
        blue_events,
        blue_launched_long,
        blue_launched_short,
        blue_attempted,
        blue_sel_long,
        blue_sel_short,
        blue_sel_target,
        blue_attempted_long_matrix,
        blue_attempted_short_matrix,
        blue_selected_long_matrix,
        blue_selected_short_matrix,
    ) = _queue_and_collect_launches(
        state=state,
        config=config,
        side_name="blue",
        own=state.blue,
        enemy=state.red,
        hit_targets=blue_hit_targets,
        fireable_long=blue_fireable_long,
        fireable_short=blue_fireable_short,
    )

    launched_events = red_events + blue_events
    delay = config.weapon.attack_effect_delay
    due_events: list[MissileEvent] = []
    if delay <= 0:
        due_events.extend(launched_events)
    else:
        state.missile_queue.extend(launched_events)

    still_pending: list[MissileEvent] = []
    for event in state.missile_queue:
        if event.resolve_step <= state.step_count:
            due_events.append(event)
        else:
            still_pending.append(event)
    state.missile_queue = still_pending

    resolved_records, killed_red, killed_blue, missiles_hit, missiles_missed = _resolve_due_events(
        state=state,
        due_events=due_events,
        hit_prob_enable=config.weapon.hit_prob_enable,
    )

    launch_records = red_launch_records + blue_launch_records
    return WeaponStepResult(
        red_fireable_long=red_fireable_long,
        red_fireable_short=red_fireable_short,
        blue_fireable_long=blue_fireable_long,
        blue_fireable_short=blue_fireable_short,
        launch_records=launch_records,
        resolved_records=resolved_records,
        red_valid_fire=red_valid,
        red_invalid_fire=red_invalid,
        blue_valid_fire=blue_valid,
        blue_invalid_fire=blue_invalid,
        killed_red=killed_red,
        killed_blue=killed_blue,
        missiles_launched_long=red_launched_long + blue_launched_long,
        missiles_launched_short=red_launched_short + blue_launched_short,
        missiles_hit=missiles_hit,
        missiles_missed=missiles_missed,
        red_attempted_fire=red_attempted,
        blue_attempted_fire=blue_attempted,
        red_selected_long=red_sel_long,
        red_selected_short=red_sel_short,
        blue_selected_long=blue_sel_long,
        blue_selected_short=blue_sel_short,
        red_selected_target_idx=red_sel_target,
        blue_selected_target_idx=blue_sel_target,
        red_attempted_long_matrix=red_attempted_long_matrix,
        red_attempted_short_matrix=red_attempted_short_matrix,
        blue_attempted_long_matrix=blue_attempted_long_matrix,
        blue_attempted_short_matrix=blue_attempted_short_matrix,
        red_selected_long_matrix=red_selected_long_matrix,
        red_selected_short_matrix=red_selected_short_matrix,
        blue_selected_long_matrix=blue_selected_long_matrix,
        blue_selected_short_matrix=blue_selected_short_matrix,
    )
