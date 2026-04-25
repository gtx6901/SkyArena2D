from __future__ import annotations

from skyarena2d.core.config import EnvConfig
from skyarena2d.core.engine import SkyArenaEngine
from skyarena2d.core.termination import check_termination


def _env() -> SkyArenaEngine:
    cfg = EnvConfig()
    cfg.teams.red_fighters = 2
    cfg.teams.blue_fighters = 2
    cfg.spawn.jitter = 0.0
    cfg.max_steps = 5
    env = SkyArenaEngine(cfg)
    env.reset(seed=0)
    return env


def test_elimination_ends_episode() -> None:
    env = _env()
    state = env.get_state()
    state.blue.alive[:] = False
    term = check_termination(state, env.config)
    assert term.done
    assert term.winner == "red"


def test_ammo_depleted_ends_episode() -> None:
    env = _env()
    state = env.get_state()
    state.red.long_ammo[:] = 0
    state.red.short_ammo[:] = 0
    state.blue.long_ammo[:] = 0
    state.blue.short_ammo[:] = 0
    term = check_termination(state, env.config)
    assert term.done
    assert term.reason == "ammo_depleted"


def test_max_steps_ends_episode() -> None:
    env = _env()
    state = env.get_state()
    state.step_count = env.config.max_steps
    term = check_termination(state, env.config)
    assert term.done
    assert term.reason == "max_steps"


def test_survival_count_decides_winner() -> None:
    env = _env()
    state = env.get_state()
    state.red.long_ammo[:] = 0
    state.red.short_ammo[:] = 0
    state.blue.long_ammo[:] = 0
    state.blue.short_ammo[:] = 0
    state.blue.alive[0] = False
    term = check_termination(state, env.config)
    assert term.winner == "red"
