import numpy as np
import pytest
import torch

from skyarena2d.rl.algo.ppo_utils import (
    ValueNorm,
    chunk_initial_states,
    clipped_value_loss,
    gather_recurrent_chunks,
    linear_lr,
    make_recurrent_chunks,
)


def test_recurrent_chunks_do_not_cross_episode_boundaries() -> None:
    done = np.array([[False], [True], [False], [False], [True]])
    chunks = make_recurrent_chunks(done, num_agents=1, chunk_length=3)
    assert [(x.start, x.stop) for x in chunks] == [(0, 2), (2, 5)]

    values = np.arange(5, dtype=np.float32).reshape(5, 1, 1)
    packed, valid = gather_recurrent_chunks(values, chunks)
    np.testing.assert_array_equal(packed, [[0, 1, 0], [2, 3, 4]])
    np.testing.assert_array_equal(valid, [[True, True, False], [True, True, True]])

    h = np.arange(10, dtype=np.float32).reshape(5, 1, 1, 2)
    initial_h, initial_c = chunk_initial_states(h, h + 100, chunks)
    np.testing.assert_array_equal(initial_h, [[0, 1], [4, 5]])
    np.testing.assert_array_equal(initial_c, [[100, 101], [104, 105]])


def test_value_clipping_uses_worse_of_clipped_and_unclipped_error() -> None:
    value = torch.tensor([3.0, 1.1])
    old_value = torch.tensor([0.0, 1.0])
    returns = torch.tensor([2.0, 1.0])
    # First sample: clipped prediction 0.2 is worse (error 1.8) than prediction 3 (error 1).
    expected = 0.5 * torch.tensor([1.8**2, 0.1**2]).mean()
    assert torch.allclose(clipped_value_loss(value, old_value, returns, 0.2), expected)


def test_value_norm_round_trip_and_linear_schedule() -> None:
    normalizer = ValueNorm()
    values = torch.tensor([1.0, 2.0, 3.0])
    normalizer.update(values)
    normalized = normalizer.normalize(values)
    assert torch.allclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(normalizer.denormalize(normalized), values, atol=1e-5)
    assert linear_lr(3e-4, 50, 100, final_lr=1e-4) == pytest.approx(2e-4)
    assert linear_lr(3e-4, 200, 100, final_lr=1e-4) == pytest.approx(1e-4)
