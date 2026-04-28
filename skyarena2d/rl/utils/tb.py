"""TensorBoard utilities for SkyArena MAPPO training."""
from __future__ import annotations

from numbers import Number
from typing import Any, Dict, Optional

from .checkpoint import tensorboard_dir


def build_writer(train_cfg: dict, purge_step: Optional[int] = None):
    """Build TensorBoard SummaryWriter if tensorboard logging is enabled."""
    logging_cfg = train_cfg.get("logging", {})
    if isinstance(logging_cfg, dict) and not bool(logging_cfg.get("tensorboard", True)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[tensorboard] SummaryWriter not available, skipping.", flush=True)
        return None
    tb_dir = tensorboard_dir(train_cfg)
    tb_dir.mkdir(parents=True, exist_ok=True)
    if purge_step is not None and int(purge_step) >= 0:
        # Use filename_suffix to create a fresh events file on resume,
        # avoiding conflicts with stale data in old event files that
        # purge_step alone may not reliably clean across PyTorch versions.
        import time as _time
        suffix = f"_resume_{int(_time.time())}"
        print("[tensorboard] log_dir=%s purge_step=%d suffix=%s" % (tb_dir, int(purge_step), suffix), flush=True)
        return SummaryWriter(log_dir=str(tb_dir), purge_step=int(purge_step), filename_suffix=suffix)
    return SummaryWriter(log_dir=str(tb_dir))


def log_scalars(writer, prefix: str, values: Dict[str, Any], step: int) -> None:
    if writer is None:
        return
    for key, value in values.items():
        if not isinstance(value, Number):
            continue
        writer.add_scalar(f"{prefix}/{key}", float(value), step)
    writer.flush()
