---
name: skyarena2d-cleanup
description: Use in SkyArena2D for no-semantics cleanup, unused imports, placeholder comments, obsolete helpers, small docs sync, or refactors that must not alter obs/action/reward/PPO behavior.
---

# SkyArena2D No-Semantics Cleanup

Use this skill for narrow cleanup only.

## Required Context

Read these first:

- `AGENTS.md`
- `docs/agent_workflows/cleanup.md`
- `docs/CODE_MAP.md`

## Rules

- Do not change observation shape, action semantics, PPO, GAE, reward, weapon semantics, recurrent hidden state, actor, critic, env core, opponents, or key diagnostics.
- Locate all references before deleting or renaming.
- Keep cleanup separate from behavior fixes.
- Avoid formatting churn.

## Expected Output

- Files changed.
- Why behavior is unchanged.
- Tests run and result.
