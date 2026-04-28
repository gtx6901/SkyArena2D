---
name: skyarena2d-action-semantics
description: Use in SkyArena2D when reviewing target/fire action semantics, target masks, fireability, rollout/eval action consistency, adapter_zeroed_fire_rate, or target_selected_nonfireable_rate.
---

# SkyArena2D Action Semantics

Use this skill for target/fire action audits.

## Required Context

Read these first:

- `AGENTS.md`
- `docs/agent_workflows/action_semantics_audit.md`
- `docs/CODE_MAP.md`

## Rules

- Do not modify PPO, reward, actor, critic, env core, opponents, or action space dimensions.
- Treat `target_action` as `fire_target_action`.
- `target_action == 0` means no target.
- `target_action > 0` may select only a fireable candidate.
- `fireable = candidate_can_long || candidate_can_short`.
- Visible/tracked non-fireable candidates may appear in entity features but must not be selectable by target action.

## Expected Output

- State whether rollout, PPO inputs, eval, and metrics share one action semantic.
- Report `adapter_zeroed_fire_rate` and `target_selected_nonfireable_rate` when available.
- Include tests or smoke commands run.
