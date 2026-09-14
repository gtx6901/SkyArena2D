"""Opponent sampling for curriculum and robustness training."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock

import numpy as np

from .base import BaseRuleOpponent


@dataclass(slots=True)
class OpponentRecord:
    games: int = 0
    red_score: float = 0.0

    @property
    def red_win_rate(self) -> float:
        return self.red_score / self.games if self.games else 0.5


class RuleOpponentPool:
    """Sample rule opponents by estimated learning value.

    Unseen opponents receive uniform priority. Afterwards the default weight
    peaks near a 50% red score, where an opponent is neither solved nor
    hopeless. A small uniform mixture prevents any opponent from disappearing.
    ``fix_rule_v2`` can therefore remain an anchor while easier rules provide
    curriculum pressure.
    """

    def __init__(
        self,
        names: list[str],
        registry: Mapping[str, type[BaseRuleOpponent]],
        *,
        seed: int = 0,
        uniform_mix: float = 0.15,
        kwargs_by_name: Mapping[str, dict] | None = None,
    ) -> None:
        if not names:
            raise ValueError("opponent pool cannot be empty")
        unknown = [name for name in names if name not in registry]
        if unknown:
            raise ValueError(f"unknown opponents: {unknown}")
        self.names = list(dict.fromkeys(names))
        self.registry = registry
        self.uniform_mix = float(np.clip(uniform_mix, 0.0, 1.0))
        self.kwargs_by_name = dict(kwargs_by_name or {})
        self.rng = np.random.default_rng(seed)
        self.records = {name: OpponentRecord() for name in self.names}
        self._lock = RLock()

    def probabilities(self) -> np.ndarray:
        with self._lock:
            learning_value = np.asarray(
                [
                    1.0 if self.records[name].games == 0
                    else max(
                        self.records[name].red_win_rate
                        * (1.0 - self.records[name].red_win_rate),
                        1e-3,
                    )
                    for name in self.names
                ],
                dtype=np.float64,
            )
            adaptive = learning_value / learning_value.sum()
            uniform = np.full(len(self.names), 1.0 / len(self.names), dtype=np.float64)
            return (1.0 - self.uniform_mix) * adaptive + self.uniform_mix * uniform

    def sample(self, *, seed: int | None = None) -> tuple[str, BaseRuleOpponent]:
        name = self.sample_name()
        kwargs = dict(self.kwargs_by_name.get(name, {}))
        kwargs["seed"] = seed
        return name, self.registry[name](**kwargs)

    def sample_name(self) -> str:
        """Sample only the rule name, for remotely hosted environments."""
        with self._lock:
            index = int(self.rng.choice(len(self.names), p=self.probabilities()))
        return self.names[index]

    def record_result(self, name: str, winner: str) -> None:
        if name not in self.records:
            raise KeyError(name)
        with self._lock:
            record = self.records[name]
            record.games += 1
            if winner == "red":
                record.red_score += 1.0
            elif winner not in ("blue", "draw", "ongoing"):
                raise ValueError(f"unknown winner: {winner}")
            elif winner == "draw":
                record.red_score += 0.5

    def state_dict(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {
                name: {"games": record.games, "red_score": record.red_score}
                for name, record in self.records.items()
            }

    def load_state_dict(self, state: Mapping[str, Mapping[str, float | int]]) -> None:
        with self._lock:
            for name, values in state.items():
                if name in self.records:
                    self.records[name].games = int(values.get("games", 0))
                    self.records[name].red_score = float(values.get("red_score", 0.0))
