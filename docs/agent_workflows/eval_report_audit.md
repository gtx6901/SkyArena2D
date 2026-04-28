# Eval Report Audit

Use this workflow when evaluating a training run, comparing checkpoints, or reviewing an eval JSON/report.

## Scope

- Audit report meaning and consistency.
- Do not change training semantics.
- Do not alter reward, PPO, action space, actor, critic, env core, or opponents.

## Checklist

- Check that `summary` and per-episode records use the same definitions.
- Distinguish step-mean metrics from episode-final metrics.
- Treat `*_mean` as step-mean unless the report documents otherwise.
- Treat cumulative fields such as kills, missiles, and final ammo as episode-final.
- Check missiles launched, selected edges, ammo spent, kills, and losses for self-consistency.
- Check that eval seeds differ between episodes, usually `base_seed + seed_offset + episode_idx`.
- Check deterministic policy eval does not mean identical reset seeds.

## Consistency Checks

- `red_selected_edges` should align with missile launch counts.
- Ammo decrease should not exceed launches by weapon type.
- Kills should not exceed enemy units.
- `selected_expected_exchange` should be interpreted together with selected launches and fireability.
- Win rate should be read from eval, not inferred from rollout windows.

## Output

- State whether the report is internally consistent.
- List inconsistent fields and likely source.
- State whether the report supports a training/config change.
