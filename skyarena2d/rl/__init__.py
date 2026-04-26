"""Experimental MAPPO training stack for SkyArena2D.

This package is runnable for smoke and integration checks, but remains
experimental for convergence-sensitive workloads.
"""

from .adapters.skyarena_mappo_env import SkyArenaMAPPOEnv
from .algo.mappo_trainer import SkyArenaMAPPOTrainer

__all__ = ["SkyArenaMAPPOEnv", "SkyArenaMAPPOTrainer"]
