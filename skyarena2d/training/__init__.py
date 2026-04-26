"""Training-side adapters and observation builders.

This package is the policy interface layer between SkyArena core state and
the experimental RL stack.
"""

from .action_adapter import SkyArenaActionAdapter
from .obs_builder import SkyArenaTrainingObsBuilder

__all__ = ["SkyArenaActionAdapter", "SkyArenaTrainingObsBuilder"]
