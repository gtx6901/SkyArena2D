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


def test_eval_reset_seed_sequence_is_reproducible():
    cfg = load_mappo_config("configs/mappo_skyarena_smoke.yaml")
    env_a = SkyArenaMAPPOEnv(cfg, seed_offset=0, deterministic_reset=True)
    env_b = SkyArenaMAPPOEnv(cfg, seed_offset=0, deterministic_reset=True)

    env_a.reset()
    a_pos_1 = env_a.engine.state.red.pos.copy()
    env_a.reset()
    a_pos_2 = env_a.engine.state.red.pos.copy()

    env_b.reset()
    b_pos_1 = env_b.engine.state.red.pos.copy()
    env_b.reset()
    b_pos_2 = env_b.engine.state.red.pos.copy()

    assert not np.allclose(a_pos_1, a_pos_2)
    assert np.allclose(a_pos_1, b_pos_1)
    assert np.allclose(a_pos_2, b_pos_2)
