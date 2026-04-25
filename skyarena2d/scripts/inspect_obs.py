from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyarena2d.core.config import load_config
from skyarena2d.core.engine import SkyArenaEngine


def summarize_modern(side_modern: dict) -> dict:
    return {
        "self": tuple(side_modern["self"].shape),
        "allies": tuple(side_modern["allies"].shape),
        "enemies": tuple(side_modern["enemies"].shape),
        "global_state": tuple(side_modern["global_state"].shape),
        "visible_matrix": tuple(side_modern["visible_matrix"].shape),
        "fireable_long": tuple(side_modern["fireable_long"].shape),
        "fireable_short": tuple(side_modern["fireable_short"].shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_10v10_full.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    env = SkyArenaEngine(config)
    obs, info = env.reset(seed=0)

    print("=== MaCA-like RAW obs (red) sample ===")
    pprint(obs["red"]["raw"])
    print("=== MaCA-like RAW obs (blue) sample ===")
    pprint(obs["blue"]["raw"])

    print("=== Modern obs summary (red) ===")
    pprint(summarize_modern(obs["red"]["modern"]))
    print("=== Modern obs summary (blue) ===")
    pprint(summarize_modern(obs["blue"]["modern"]))
    print("=== Metrics ===")
    pprint(info["metrics"])
    env.close()


if __name__ == "__main__":
    main()
