"""Compatibility adapters between core state and legacy/raw interfaces."""

from .action_decoder import DecodedSideAction, decode_maca_side_action
from .maca_compat import build_maca_raw_obs
from .modern_obs_builder import build_modern_obs

__all__ = [
    "DecodedSideAction",
    "decode_maca_side_action",
    "build_maca_raw_obs",
    "build_modern_obs",
]
