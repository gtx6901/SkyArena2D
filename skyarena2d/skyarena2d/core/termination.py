from __future__ import annotations

from dataclasses import dataclass

from .config import EnvConfig
from .state import EnvState


@dataclass(slots=True)
class TerminationResult:
    done: bool
    truncated: bool
    reason: str
    winner: str


def _winner_by_alive(state: EnvState) -> str:
    red_alive = state.red.alive_count
    blue_alive = state.blue.alive_count
    if red_alive > blue_alive:
        return "red"
    if blue_alive > red_alive:
        return "blue"
    return "draw"


def check_termination(state: EnvState, config: EnvConfig) -> TerminationResult:
    red_alive = state.red.alive_count
    blue_alive = state.blue.alive_count

    if red_alive == 0 and blue_alive == 0:
        return TerminationResult(done=True, truncated=False, reason="mutual_elimination", winner="draw")
    if red_alive == 0:
        return TerminationResult(done=True, truncated=False, reason="red_eliminated", winner="blue")
    if blue_alive == 0:
        return TerminationResult(done=True, truncated=False, reason="blue_eliminated", winner="red")

    missiles_empty = state.red.remaining_missiles() == 0 and state.blue.remaining_missiles() == 0
    if missiles_empty and not state.missile_queue:
        return TerminationResult(
            done=True,
            truncated=False,
            reason="ammo_depleted",
            winner=_winner_by_alive(state),
        )

    if state.step_count >= config.max_steps:
        return TerminationResult(
            done=True,
            truncated=True,
            reason="max_steps",
            winner=_winner_by_alive(state),
        )

    return TerminationResult(done=False, truncated=False, reason="", winner="ongoing")
