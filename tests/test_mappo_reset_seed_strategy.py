from __future__ import annotations

import numpy as np

from skyarena2d.rl.adapters.skyarena_mappo_env import SkyArenaMAPPOEnv
from skyarena2d.rl.utils.config import load_mappo_config


def test_training_reset_seed_changes_across_episodes():
    cfg = load_mappo_config("configs/mappo_skyarena_smoke.yaml")
    env = SkyArenaMAPPOEnv(cfg, seed_offset=0, deterministic_reset=False)

    env.reset()
    pos_1 = env.engine.state.red.pos.copy()
    env.reset()
    pos_2 = env.engine.state.red.pos.copy()

    assert not np.allclose(pos_1, pos_2)


def test_eval_reset_can_be_deterministic():
    cfg = load_mappo_config("configs/mappo_skyarena_smoke.yaml")
    env = SkyArenaMAPPOEnv(cfg, seed_offset=0, deterministic_reset=True)

    env.reset()
    pos_1 = env.engine.state.red.pos.copy()
    env.reset()
    pos_2 = env.engine.state.red.pos.copy()

    assert np.allclose(pos_1, pos_2)
