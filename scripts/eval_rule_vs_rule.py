from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyarena2d.core.config import load_config
from skyarena2d.core.engine import SkyArenaEngine
from skyarena2d.logging.episode_summary import EpisodeSummary
from skyarena2d.logging.trace_recorder import TraceRecorder
from skyarena2d.opponents import RULES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--red", default="rush_rule")
    parser.add_argument("--blue", default="patrol_rule")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--config", default="configs/env_10v10_full.yaml")
    parser.add_argument("--record_trace", action="store_true", help="Record per-step trace JSONL files")
    parser.add_argument("--trace_dir", type=str, default="logs/traces", help="Directory for trace JSONL files")
    parser.add_argument("--record_summary", action="store_true", help="Record episode summary JSON")
    parser.add_argument("--summary_path", type=str, default="logs/summary.json", help="Path for summary JSON output")
    args = parser.parse_args()

    if args.red not in RULES:
        raise ValueError(f"Unknown red rule: {args.red}")
    if args.blue not in RULES:
        raise ValueError(f"Unknown blue rule: {args.blue}")

    config = load_config(args.config)
    env = SkyArenaEngine(config)
    red_rule = RULES[args.red]()
    blue_rule = RULES[args.blue]()

    wins = {"red": 0, "blue": 0, "draw": 0}
    returns_red: list[float] = []
    returns_blue: list[float] = []
    red_kills: list[int] = []
    blue_kills: list[int] = []
    metrics_acc: list[dict[str, float | int | None]] = []
    all_summaries: list[dict] = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        red_rule.reset(ep)
        blue_rule.reset(ep + 10000)

        done = False
        ret_red = 0.0
        ret_blue = 0.0
        last_info = None

        recorder: TraceRecorder | None = None
        if args.record_trace:
            recorder = TraceRecorder(trace_dir=args.trace_dir, episode=ep)

        ep_summary: EpisodeSummary | None = None
        if args.record_summary:
            ep_summary = EpisodeSummary()

        while not done:
            state = env.get_state()
            red_action = red_rule.act(obs["red"], "red", state.step_count)
            blue_action = blue_rule.act(obs["blue"], "blue", state.step_count)
            obs, reward, done, _, info = env.step({"red": red_action, "blue": blue_action})
            ret_red += float(reward["red"])
            ret_blue += float(reward["blue"])
            last_info = info

            # Trace recording uses state.cache (populated after env.step)
            if recorder is not None or ep_summary is not None:
                post_state = env.get_state()
                metrics = info.get("metrics", {})
                reward_components = info.get("reward_components", {})

                if recorder is not None:
                    recorder.record_step(post_state, metrics, reward_components)

                if ep_summary is not None:
                    ep_summary.record_step(metrics, reward_components)

        if recorder is not None:
            recorder.close()

        assert last_info is not None
        winner = str(last_info["winner"])
        wins[winner] = wins.get(winner, 0) + 1
        returns_red.append(ret_red)
        returns_blue.append(ret_blue)
        red_kills.append(int(last_info["metrics"]["red_kills"]))
        blue_kills.append(int(last_info["metrics"]["blue_kills"]))
        metrics_acc.append(last_info["metrics"])

        if ep_summary is not None:
            final_state = env.get_state()
            summary = ep_summary.finalize(final_state, last_info)
            all_summaries.append(summary)

        print(
            f"episode={ep + 1} winner={winner} red_return={ret_red:.2f} blue_return={ret_blue:.2f} red_kills={red_kills[-1]} blue_kills={blue_kills[-1]}"
        )

    def avg_metric(key: str) -> float:
        values = [float(m[key]) for m in metrics_acc if m.get(key) is not None]
        return float(np.mean(values)) if values else 0.0

    print("=== Summary ===")
    print("wins:", wins)
    print("avg_red_return:", float(np.mean(returns_red)) if returns_red else 0.0)
    print("avg_blue_return:", float(np.mean(returns_blue)) if returns_blue else 0.0)
    print("avg_red_kills:", float(np.mean(red_kills)) if red_kills else 0.0)
    print("avg_blue_kills:", float(np.mean(blue_kills)) if blue_kills else 0.0)
    print("avg_fireability_edge_advantage:", avg_metric("fireability_edge_advantage"))
    print("avg_expected_exchange_proxy:", avg_metric("expected_exchange_proxy"))
    print("avg_contact_to_fire_gap:", avg_metric("contact_to_fire_gap"))

    if args.record_summary and all_summaries:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"Summary written to {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
