---
name: skyarena2d-invalid-fire
description: Use in SkyArena2D when red_invalid_fire_count is high, selected launches are rejected, candidate_can_long/short may be stale, or fireability differs between policy obs and engine.step.
---

# SkyArena2D Invalid Fire Audit

Use this skill to locate invalid fire causes before changing reward or config.

## Required Context

Read these first:

- `AGENTS.md`
- `docs/agent_workflows/invalid_fire_audit.md`
- `docs/agent_workflows/action_semantics_audit.md`
- `docs/CODE_MAP.md`

## Rules

- First pass is diagnosis only.
- Do not modify PPO, reward, actor, critic, env core, opponents, action space, or weapon semantics.
- Compare policy obs fireability against fireability recomputed in `engine.step`.
- Check ammo, visibility, target alive, range edges, candidate staleness, and target index decoding.

## Expected Output

- Invalid fire reason distribution if possible.
- Earliest path where semantics diverge.
- Recommended smallest test or diagnostic.
