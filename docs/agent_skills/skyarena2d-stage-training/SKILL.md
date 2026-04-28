---
name: skyarena2d-stage-training
description: Use in SkyArena2D when running or reviewing Stage0/Stage1/Stage2/Stage3/Stage4 MAPPO training, warm starts, smoke training, eval smoke, or curriculum readiness.
---

# SkyArena2D Stage Training

Use this skill for staged MAPPO training and evaluation workflows.

## Required Context

Read these first:

- `AGENTS.md`
- `docs/agent_workflows/stage_training.md`
- `docs/TRAINING.md`
- `docs/CONFIG_GUIDE.md`

## Rules

- Do not run CUDA long training unless the user explicitly asks.
- Do not modify training code, PPO, reward, actor, critic, env core, opponents, or action space.
- Use project Python environment; do not create a virtual environment or install global dependencies.
- Logs should go under `/tmp/skyarena_logs`.

## Expected Output

- Commands run.
- Device, config, total steps, eval interval, checkpoint/warm-start behavior.
- Test and smoke result.
- Artifact locations.
