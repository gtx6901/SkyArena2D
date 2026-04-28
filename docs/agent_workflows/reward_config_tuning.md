# Reward And Config Tuning

Use this workflow for small reward or stage config changes after diagnostics show a specific training bottleneck.

## Gate

- Change reward or config only with evidence from eval reports, TensorBoard metrics, traces, or focused tests.
- Prefer stage YAML changes over code changes.
- Do not change more than one coefficient family at once.
- Do not tune from a single noisy rollout metric.

## Allowed First Moves

- Adjust a stage YAML value with a clear hypothesis.
- Change training duration, eval interval, or stage opponent only when it matches the curriculum plan.
- Add a focused diagnostic or pytest before changing semantics.

## Avoid

- Do not add contact, closing, formation, tracking, or similar rule-flavored rewards.
- Do not hide a target/fire semantic bug behind reward shaping.
- Do not change PPO, GAE, advantage normalization, actor, critic, action space, env core, or opponents.
- Do not mix reward tuning with cleanup.

## Required Comparison

After a change, compare before and after:

- config diff
- eval report
- `red_fireable_edges`
- `red_attempted_edges`
- `red_selected_edges`
- `red_invalid_fire_count`
- `selected_expected_exchange`
- `eval/win_rate`

## Output

- State the evidence, hypothesis, changed value, and expected metric movement.
- Include the exact validation command.
