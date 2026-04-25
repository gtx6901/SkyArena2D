"""Config loader for SkyArena MAPPO training."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_mappo_config(path: str | Path) -> Dict[str, Any]:
    """Load MAPPO config from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def require_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    """Get required config section, raising if missing."""
    if section not in cfg:
        raise KeyError(f"Missing required config section: {section}")
    return cfg[section]
