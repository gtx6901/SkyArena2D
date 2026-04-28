---
name: skyarena2d-eval-report
description: Use in SkyArena2D when auditing eval reports, win_rate, episode summaries, step_mean versus episode_final metrics, missiles/ammo/kills consistency, or eval seed behavior.
---

# SkyArena2D Eval Report Audit

Use this skill for read-only eval report interpretation.

## Required Context

Read these first:

- `AGENTS.md`
- `docs/agent_workflows/eval_report_audit.md`
- `docs/TRAINING.md`

## Rules

- Do not change training semantics while reviewing a report.
- Separate step-mean metrics from episode-final metrics.
- Check summary and episode records for matching definitions.
- Check eval seeds differ between episodes unless a deterministic reset test explicitly says otherwise.

## Expected Output

- State whether the report is internally consistent.
- List inconsistent fields and likely cause.
- Say whether the report supports reward/config tuning.
