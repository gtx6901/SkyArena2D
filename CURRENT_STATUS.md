# Current Status

## Scope

This document is a concise, code-aligned status snapshot.

## What Is Runnable

- Environment reset/step and rule-vs-rule evaluation are runnable.
- GUI rendering and scoreboard are runnable.
- MAPPO smoke training is runnable.
- MAPPO long training pipeline is available but experimental.

## Reliability Boundary

Stable:
- core environment execution pipeline
- rule opponents and rule-vs-rule scripts
- basic renderer and trace logging

Experimental:
- skyarena2d/rl training stack
- recurrent PPO updates and search goal manager
- long-horizon convergence behavior

## Current Recommendation

- Treat MAPPO as experimental-but-runnable.
- Use smoke configs for integration checks.
- Keep environment and reward semantics stable while tuning.
