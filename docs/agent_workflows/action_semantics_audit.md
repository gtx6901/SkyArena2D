# Action Semantics Audit

Use this workflow when `target_action`, `fire_action`, target masks, or launch diagnostics look inconsistent.

## Scope

- Audit semantics across obs building, rollout collection, PPO update inputs, eval, and metrics.
- Do not modify PPO, reward, actor, critic, or model shape.
- Prefer diagnosis and tests before any behavior change.

## Checklist

- Confirm `target_mask` allows only fireable candidates.
- Confirm `fireable = candidate_can_long || candidate_can_short`.
- Confirm visible or tracked non-fireable candidates may appear in entity features but cannot be selected by `target_action`.
- Confirm rollout, PPO update, policy eval, and GUI eval use the same action semantics.
- Confirm fire is resolved against the current observed state before this step's movement produces the next obs.
- Check `target_selected_nonfireable_rate`; it should be near 0.
- Check `adapter_zeroed_fire_rate`; it must be 0.
- Check `fire_action > 0` corresponds to an actual legal launch, not just a target candidate.
- Compare `red_attempted_edges` and `red_selected_edges` to separate attempts from accepted launches.

## Evidence To Collect

- Relevant obs/action adapter code paths.
- Metrics for `target_action_nonzero_rate`, `fire_action_nonzero_rate`, `red_fireable_edges`, `red_attempted_edges`, `red_selected_edges`, and `red_invalid_fire_count`.
- Narrow pytest or smoke output if a behavior contract is changed.

## Output

- State whether action semantics are consistent.
- List any mismatched path with file and line reference.
- Recommend the smallest next step.
