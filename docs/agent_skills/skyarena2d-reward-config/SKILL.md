---
name: skyarena2d-reward-config
description: Use in SkyArena2D for small reward or stage YAML tuning after diagnostics support it, especially when comparing eval reports and avoiding rule-flavored reward shaping.
---

# SkyArena2D Reward And Config Tuning

Use this skill only after evidence points to a specific reward/config bottleneck.

## Required Context

Read these first:

- `AGENTS.md`
- `docs/agent_workflows/reward_config_tuning.md`
- `docs/CONFIG_GUIDE.md`
- `docs/TRAINING.md`

## Rules

- Prefer stage YAML changes over code changes.
- Change only one coefficient family at a time.
- Do not add contact, closing, formation, tracking, or similar rule-flavored rewards.
- Do not change PPO, GAE, actor, critic, action space, env core, opponents, or weapon semantics.

## Expected Output

- Evidence, hypothesis, changed value, expected metric movement.
- Before/after eval report comparison plan.
- Validation commands.
