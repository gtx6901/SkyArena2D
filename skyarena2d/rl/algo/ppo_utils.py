"""Small, reusable numerical utilities for recurrent PPO learners."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class RecurrentChunk:
    """A contiguous sequence for one environment-agent trajectory."""

    start: int
    stop: int
    env_index: int
    agent_index: int

    @property
    def length(self) -> int:
        return self.stop - self.start


def make_recurrent_chunks(
    done: np.ndarray,
    num_agents: int,
    chunk_length: int,
    *,
    drop_last: bool = False,
) -> list[RecurrentChunk]:
    """Split ``[T,E]``/``[T,E,N]`` rollouts without crossing terminations."""
    done_array = np.asarray(done, dtype=bool)
    if done_array.ndim not in (2, 3):
        raise ValueError("done must be [T,E] or [T,E,N]")
    if num_agents <= 0 or chunk_length <= 0:
        raise ValueError("num_agents and chunk_length must be positive")
    if done_array.ndim == 2:
        done_array = np.broadcast_to(done_array[..., None], (*done_array.shape, num_agents))
    elif done_array.shape[-1] != num_agents:
        raise ValueError("done agent axis does not match num_agents")

    rollout_steps, num_envs, _ = done_array.shape
    chunks: list[RecurrentChunk] = []
    for env_index in range(num_envs):
        for agent_index in range(num_agents):
            segment_start = 0
            terminal_steps = np.flatnonzero(done_array[:, env_index, agent_index])
            for segment_stop in [*(terminal_steps + 1).tolist(), rollout_steps]:
                if segment_stop <= segment_start:
                    continue
                for start in range(segment_start, segment_stop, chunk_length):
                    stop = min(start + chunk_length, segment_stop)
                    if not drop_last or stop - start == chunk_length:
                        chunks.append(RecurrentChunk(start, stop, env_index, agent_index))
                segment_start = segment_stop
    return chunks


def iter_recurrent_minibatches(
    chunks: Sequence[RecurrentChunk],
    minibatch_chunks: int,
    *,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
) -> Iterator[list[RecurrentChunk]]:
    """Yield chunk groups; randomness is injectable for reproducible updates."""
    if minibatch_chunks <= 0:
        raise ValueError("minibatch_chunks must be positive")
    indices = np.arange(len(chunks))
    if shuffle:
        (rng or np.random.default_rng()).shuffle(indices)
    for start in range(0, len(indices), minibatch_chunks):
        yield [chunks[index] for index in indices[start : start + minibatch_chunks]]


def gather_recurrent_chunks(
    values: np.ndarray,
    chunks: Sequence[RecurrentChunk],
    *,
    pad_value: float | int | bool = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack ``[T,E,N,...]`` values and return a valid-timestep mask."""
    array = np.asarray(values)
    if array.ndim < 3:
        raise ValueError("values must have leading [T,E,N] axes")
    if not chunks:
        return (
            np.empty((0, 0, *array.shape[3:]), dtype=array.dtype),
            np.empty((0, 0), dtype=bool),
        )
    max_length = max(chunk.length for chunk in chunks)
    packed = np.full((len(chunks), max_length, *array.shape[3:]), pad_value, dtype=array.dtype)
    valid = np.zeros((len(chunks), max_length), dtype=bool)
    for row, chunk in enumerate(chunks):
        if not (0 <= chunk.start < chunk.stop <= array.shape[0]):
            raise ValueError(f"chunk {chunk} lies outside the rollout")
        if chunk.env_index >= array.shape[1] or chunk.agent_index >= array.shape[2]:
            raise ValueError(f"chunk {chunk} references a missing environment or agent")
        packed[row, : chunk.length] = array[
            chunk.start : chunk.stop, chunk.env_index, chunk.agent_index
        ]
        valid[row, : chunk.length] = True
    return packed, valid


def chunk_initial_states(
    recurrent_h: np.ndarray,
    recurrent_c: np.ndarray,
    chunks: Sequence[RecurrentChunk],
) -> tuple[np.ndarray, np.ndarray]:
    """Gather the exact pre-action recurrent state for every sequence chunk."""
    if recurrent_h.shape != recurrent_c.shape or recurrent_h.ndim != 4:
        raise ValueError("recurrent_h and recurrent_c must share shape [T,E,N,H]")
    if chunks:
        h = np.stack([recurrent_h[x.start, x.env_index, x.agent_index] for x in chunks], axis=0)
        c = np.stack([recurrent_c[x.start, x.env_index, x.agent_index] for x in chunks], axis=0)
    else:
        h = np.empty((0, recurrent_h.shape[-1]), dtype=recurrent_h.dtype)
        c = np.empty((0, recurrent_c.shape[-1]), dtype=recurrent_c.dtype)
    return h, c


def linear_lr(
    initial_lr: float,
    step: int,
    total_steps: int,
    *,
    final_lr: float = 0.0,
) -> float:
    """Linearly interpolate the learning rate and clamp after the last step."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    fraction = min(max(step / total_steps, 0.0), 1.0)
    return float(initial_lr + fraction * (final_lr - initial_lr))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def clipped_value_loss(
    value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """PPO's clipped value objective, including its conventional 1/2 factor."""
    if clip_range < 0:
        raise ValueError("clip_range must be non-negative")
    clipped = old_value + (value - old_value).clamp(-clip_range, clip_range)
    loss = 0.5 * torch.maximum((value - returns).square(), (clipped - returns).square())
    if valid_mask is None:
        return loss.mean()
    weights = valid_mask.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def normalize_advantages(
    advantages: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Normalize only live/valid samples and leave padded samples at zero."""
    if valid_mask is None:
        return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + epsilon)
    valid = valid_mask.bool()
    selected = advantages[valid]
    if selected.numel() == 0:
        return torch.zeros_like(advantages)
    normalized = (advantages - selected.mean()) / (selected.std(unbiased=False) + epsilon)
    return torch.where(valid, normalized, torch.zeros_like(normalized))


class ValueNorm(nn.Module):
    """Running return normalization without PopArt output-layer adjustment.

    This is the lightweight ValueNorm option.  A learner using PopArt should
    not also apply this class to the same critic target.
    """

    def __init__(self, epsilon: float = 1e-5, clip_range: float | None = None) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.clip_range = clip_range
        self.register_buffer("mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("variance", torch.ones((), dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, values: torch.Tensor, valid_mask: torch.Tensor | None = None) -> None:
        samples = values.detach().to(dtype=torch.float64)
        if valid_mask is not None:
            samples = samples[valid_mask.bool()]
        samples = samples.reshape(-1)
        if samples.numel() == 0:
            return
        batch_mean = samples.mean()
        batch_var = samples.var(unbiased=False)
        batch_count = torch.as_tensor(samples.numel(), dtype=torch.float64, device=samples.device)
        if self.count.item() == 0:
            self.mean.copy_(batch_mean)
            self.variance.copy_(batch_var)
            self.count.copy_(batch_count)
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        merged_mean = self.mean + delta * batch_count / total
        merged_m2 = (
            self.variance * self.count
            + batch_var * batch_count
            + delta.square() * self.count * batch_count / total
        )
        self.mean.copy_(merged_mean)
        self.variance.copy_(merged_m2 / total)
        self.count.copy_(total)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        result = (values - self.mean.to(values.dtype)) / torch.sqrt(
            self.variance.to(values.dtype) + self.epsilon
        )
        if self.clip_range is not None:
            result = result.clamp(-self.clip_range, self.clip_range)
        return result

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * torch.sqrt(self.variance.to(values.dtype) + self.epsilon) + self.mean.to(values.dtype)
