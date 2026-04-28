# Invalid Fire Audit

Use this workflow when `red_invalid_fire_count` is high or legal-looking launches are rejected.

## Scope

- Locate why invalid fire occurs.
- Do not rush into reward changes.
- Do not modify PPO, actor, critic, env core, or opponents during the first audit pass.

## Checklist

- Check whether `candidate_can_long` and `candidate_can_short` are stale.
- Compare policy observation fireability with the fireability recomputed inside `engine.step`.
- Confirm both values refer to the same step moment.
- Inspect ammo state, visibility, target alive status, range edge cases, and pending missile timing.
- Separate invalid long and invalid short cases if the diagnostics allow it.
- Check whether target indices are shifted or decoded differently between adapter and engine.

## Reason Distribution

Produce an invalid fire reason distribution when possible:

- no ammo
- target not visible
- target not alive
- target out of long range
- target out of short range
- stale candidate
- action decode or index mismatch
- other or unknown

## Output

- Report the dominant invalid fire reason.
- Identify the earliest code path where semantics diverge.
- Recommend a test or diagnostic before any reward/config adjustment.
