# SkyArena2D Agent Rules

## Repository Boundary

- Work only inside the current SkyArena2D repository.
- Treat MaCA only as historical reference.
- Do not synchronize, maintain, or patch the old MaCA project.
- Do not copy MaCA black-box behavior unless the local SkyArena2D contract explicitly requires it.

## Project Goal

- The training target is a red MAPPO agent using CTDE and a recurrent actor.
- The long-term objective is to beat `fix_rule_v2`.
- Red RL should learn search, maneuver, target selection, and fire decisions.
- EW remains rule-based for now and must not enter the actor action space.

## Engineering Principles

- Prefer small, verifiable, reversible changes.
- Read existing code and docs before editing.
- Do not write very long functions.
- Do not add unnecessary abstractions.
- Do not expand features for a sense of platform completeness.
- Keep behavior changes tied to beating `fix_rule_v2`.
- After a change, run the relevant smoke command or pytest target.

## Restricted Areas

Do not casually modify:

- PPO loss
- GAE
- advantage normalization
- recurrent hidden state
- actor or critic structure
- action log-prob semantics
- action space dimensions
- weapon hit semantics
- opponent behavior
- learned EW
- self-play

If one of these must change, first document the bug, the intended training semantics, checkpoint impact, and validation plan.

## Current Action Semantics

- `target_action` means `fire_target_action`.
- `target_action == 0` means no target.
- `target_action > 0` may select only a fireable candidate.
- `fireable = candidate_can_long || candidate_can_short`.
- `engine.step()` resolves weapon fire against the current observed state before applying this step's movement.
- Visible or tracked candidates may appear in `entity_features`, but are not necessarily selectable by `target_action`.
- `adapter_zeroed_fire_rate` must be 0.
- `target_selected_nonfireable_rate` should be near 0.

## Debug Priority

When training or eval behavior looks wrong, inspect these first:

- `red_fireable_edges`
- `red_attempted_edges`
- `red_selected_edges`
- `red_invalid_fire_count`
- `target_action_nonzero_rate`
- `fire_action_nonzero_rate`
- `red_missiles_remaining`
- `selected_expected_exchange`
- `eval/win_rate`

Do not diagnose training only from PPO losses or rollout win rate.

## Tests And Artifacts

- Permanent pytest files are allowed when they lock down behavior.
- Remove temporary test files before finishing.
- Do not commit `train_dir`, checkpoints, TensorBoard event files, eval reports, GUI frames, videos, or trace dumps.
- Put local logs under `/tmp/skyarena_logs` or another ignored local directory.

## Agent Workflows

Use repo-local workflow docs for repeated tasks:

- `docs/agent_workflows/action_semantics_audit.md`
- `docs/agent_workflows/invalid_fire_audit.md`
- `docs/agent_workflows/eval_report_audit.md`
- `docs/agent_workflows/reward_config_tuning.md`
- `docs/agent_workflows/stage_training.md`
- `docs/agent_workflows/cleanup.md`

Repo-local Codex-readable skills live in `docs/agent_skills/`. When a user asks for one of these task types, read the matching `SKILL.md` first:

- `docs/agent_skills/skyarena2d-action-semantics/SKILL.md`
- `docs/agent_skills/skyarena2d-invalid-fire/SKILL.md`
- `docs/agent_skills/skyarena2d-eval-report/SKILL.md`
- `docs/agent_skills/skyarena2d-reward-config/SKILL.md`
- `docs/agent_skills/skyarena2d-stage-training/SKILL.md`
- `docs/agent_skills/skyarena2d-cleanup/SKILL.md`
