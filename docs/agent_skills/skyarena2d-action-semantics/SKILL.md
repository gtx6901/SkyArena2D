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

- Treat current action semantics as Baseline V2; Movement V3 is historical.
- Confirm the actor exposes only course, target, and fire heads.
- Treat `target_action` as engagement / attention target.
- `target_action == 0` means no target.
- `target_action > 0` may select any valid visible/tracked candidate where `entity_mask` is true.
- `fireable = candidate_can_long || candidate_can_short` gates only `fire_action`.
- `fire_action > 0` must apply only to the selected target and only when the selected target is fireable for the requested weapon.
- `course_action` decodes one of nine relative heading deltas.
- Visible/tracked non-fireable candidates may appear in entity features and may be selected by target action.
- Friendly tokens and padding must not be targetable.

## Expected Output

- State whether rollout, PPO inputs, eval, and metrics share one action semantic.
- Report `fire_nonzero_when_fireable` and accepted launch metrics when available.
- Include tests or smoke commands run.
