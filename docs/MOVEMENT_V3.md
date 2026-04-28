# Movement V3

Movement V3 replaces Movement V2 reference-frame steering with a hybrid movement-mode action space.

## Heads

The actor outputs:

- `movement_mode_logits`
- `course_logits`
- `search_goal_logits`
- `target_logits`
- `fire_logits`

`target_action` is now an engagement/attention target. Any visible or tracked `entity_mask` candidate can be selected. Fire remains strictly masked by `candidate_can_long` and `candidate_can_short`.

## Movement Modes

```text
0 FREE_COURSE
1 SEARCH_GOAL
2 NEAREST_VISIBLE
3 SELECTED_TARGET
4 INTERCEPT_TARGET
5 ORBIT_LEFT
6 ORBIT_RIGHT
7 SUPPORT
8 SEPARATION
```

`FREE_COURSE` interprets `course_action` as one of 32 absolute global heading bins.

All other modes interpret `course_action` as a residual around the mode reference. Rollout masks residual actions to:

```text
[-90, -45, -22.5, 0, 22.5, 45, 90]
```

Invalid modes are masked during rollout rather than silently repaired by the adapter.

## Search Goals

`SearchGoalManager` is a semi-MDP holder:

- The policy selects raw search goals.
- The manager decides when a goal refresh is allowed.
- On refresh, the raw policy goal is executed.
- Without refresh, the held goal continues and search-goal logprob is zero.

Actor observation uses `current_search_goal_id` with 0 as no active goal and 1..G*G as goal id + 1. The adapter receives executed region ids as 0..G*G-1.

## Compatibility

Movement V3 changes actor parameters and action log probabilities. Old Movement V2 checkpoints should not be resumed.
