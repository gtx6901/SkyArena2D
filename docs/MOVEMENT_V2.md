# Movement V2

Movement V2 changes the red policy movement interface from current-heading-relative steering to a learned reference-frame interface.

## Motivation

The old `course_action` was decoded as an offset from the current heading. A stable nonzero action therefore integrated over time and could become circling behavior. The semantic map contains target and map geometry, but the old action space still required the recurrent policy to discover a useful coordinate transform through PPO credit assignment.

Movement V2 keeps the geometry visible to the network and also exposes a better action coordinate system:

```text
heading = reference_bearing(reference_action) + course_offset(course_action)
```

The actor learns which reference to use and how to maneuver around it.

## Action Heads

The actor outputs:

- `reference_action`: movement reference frame.
- `course_action`: offset around that reference frame.
- `search_goal_action`: persistent search region.
- `target_action`: fire target slot, where 0 means no target.
- `fire_action`: no fire, long, or short.

`course_action` keeps 16 bins over the full heading circle, so the policy can still fly away, flank, cross, or orbit.

## Reference Actions

```text
0 current_heading
1 search_goal_bearing
2 selected_target_bearing
3 selected_target_intercept_bearing
4 nearest_fireable_bearing
5 ally_contact_support_bearing
6 separation_bearing
7 map_center_bearing
```

Invalid references fall back to search goal or current heading. Fallback only repairs undefined geometry; it does not select targets or fire.

## Checkpoint Compatibility

Movement V2 changes actor parameters and action log probabilities. Old Stage0 checkpoints are not a behavior baseline for Movement V2 and should not be resumed. Start a fresh Stage0 run.

## Stage0 Gate

Do not use `win_rate` alone. Inspect:

- `blue_alive_final`
- `blue_eliminated_rate`
- `episode_len`
- `heading_change_mean`
- `heading_flip_rate`
- `course_action_histogram`
- `contact_agents_mean`
- `late_contact_rate`
- `nearest_blue_distance_final`
- `search_goal_refresh_rate`
- GUI behavior
