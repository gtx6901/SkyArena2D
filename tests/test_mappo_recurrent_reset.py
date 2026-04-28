from __future__ import annotations

import numpy as np
import torch

from skyarena2d.rl.algo.mappo_trainer import SkyArenaMAPPOTrainer


def test_ppo_recurrent_hidden_reset_after_done() -> None:
    h = torch.ones((4, 3), dtype=torch.float32)
    c = torch.ones((4, 3), dtype=torch.float32) * 2.0
    prev_done = np.array([1.0, 0.0], dtype=np.float32)

    h2, c2 = SkyArenaMAPPOTrainer._reset_recurrent_hidden_after_done(h, c, prev_done, num_agents=2)

    assert torch.all(h2[:2] == 0.0)
    assert torch.all(c2[:2] == 0.0)
    assert torch.all(h2[2:] == 1.0)
    assert torch.all(c2[2:] == 2.0)
