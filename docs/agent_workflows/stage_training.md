# Stage Training

Use this workflow to run or review MAPPO curriculum stages.

## Stages

| Stage | Config | Blue rule | Purpose |
|---|---|---|---|
| 0 | `configs/mappo_skyarena_stage0_no_attack.yaml` | `no_attack_rule` | Basic pipeline, search, movement, non-combat sanity |
| 1 | `configs/mappo_skyarena_stage1_patrol.yaml` | `patrol_rule` | Low-pressure search and contact |
| 2 | `configs/mappo_skyarena_stage2_rush.yaml` | `rush_rule` | Fire-control startup under pressure |
| 3 | `configs/mappo_skyarena_stage3_fix_like.yaml` | `fix_rule_like` | Bridge to the main rule family |
| 4 | `configs/mappo_skyarena_stage4_fix_v2.yaml` | `fix_rule_v2` | Main target opponent |

## Stage Advancement

Consider moving forward only when these are stable enough for the current stage:

- `target_action_nonzero_rate` is not collapsed.
- `fire_action_nonzero_rate` is plausible when fireable edges exist.
- `red_fireable_edges` appears in eval or rollout diagnostics.
- `red_attempted_edges` and `red_selected_edges` are nonzero when expected.
- `red_invalid_fire_count` is understood and not dominating.
- `selected_expected_exchange` is improving or at least interpretable.
- `eval/win_rate` is measured with consistent eval settings.

## Warm Start

- Stage 0 starts without warm start.
- Later stages may use the previous stage checkpoint through `--init_checkpoint`.
- `resume: true` continues the same experiment and restores counters.
- `init_checkpoint` warm-starts a new stage when `resume: false`.
- If `resume: true` and `init_checkpoint` are both set, trainer behavior should be checked before trusting the run.

## Commands

Stage 0 CPU smoke:

```bash
bash scripts/smoke_stage0_cpu.sh
```

Stage 0 eval smoke:

```bash
bash scripts/eval_stage0_cpu.sh
```

Stage 0 CUDA 300k, only when explicitly requested:

```bash
bash scripts/train_stage0_300k_cuda.sh
```

## Output

- Report config, device, total steps, eval interval, checkpoint behavior, and log path.
- Do not claim convergence from smoke runs.
