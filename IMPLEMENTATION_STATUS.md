# SkyArena2D MAPPO Refactoring - Implementation Status

## Completed Components

### 1. Core State Extensions (✓ COMPLETE)
**File**: `skyarena2d/core/state.py`
- Extended `MetricsTracker` with attempted/selected tracking fields
- Extended `StepCache` with attempted/selected arrays and reward_components
- All new fields properly initialized with defaults

### 2. Weapons System Enhancement (✓ COMPLETE)
**File**: `skyarena2d/core/weapons.py`
- Extended `WeaponStepResult` with attempted/selected fields
- Modified `_queue_and_collect_launches` to track:
  - `attempted`: agent submitted fire_action > 0
  - `selected_long/short`: actually launched (attempted AND fireable)
  - `selected_target_idx`: which enemy was targeted
- Returns all new tracking arrays

### 3. Metrics System Enhancement (✓ COMPLETE)
**File**: `skyarena2d/core/metrics.py`
- Added `_selected_expected_kills()` function
- Extended `update_tracker()` to accumulate:
  - fire_attempts, fire_opportunities, fire_executions
  - invalid_fire_count, unique_selected_targets
  - selected_overkill_values
- Extended `build_metrics_snapshot()` with 15+ new metrics:
  - red/blue_attempted_edges, selected_edges
  - selected_edge_advantage
  - fire_execution_rate_given_opportunity
  - selected_expected_exchange
  - selected_overkill_mean
  - invalid_fire_count

### 4. Modular Reward System (✓ COMPLETE)
**File**: `skyarena2d/core/reward.py`
- Refactored `compute_rewards()` to support modular components
- Modules: kill_loss, valid_fire, invalid_fire, fire_execution, selected_exchange, win_loss
- Each module can be enabled/disabled via config
- Returns `RewardOutput` with `components` dict for info logging
- Supports per-module weight configuration
- Terminal win/loss only applied once (no double-counting)

### 5. Config Extension (✓ COMPLETE)
**File**: `skyarena2d/core/config.py`
- Added `reward_modules: dict` field to `EnvConfig`
- Allows YAML configuration of reward components

### 6. Fix Rule V2 Opponent (✓ COMPLETE)
**File**: `skyarena2d/opponents/fix_rule_v2.py`
- Initial push phase: red→0°, blue→180° until first contact
- Post-contact: each agent independently:
  - Locks onto nearest visible enemy
  - Fires long (prefer) or short based on distance
  - Random search when no visible enemies
- Registered in `skyarena2d/opponents/__init__.py`

## Remaining Components (NOT YET IMPLEMENTED)

### 7. Policy Observation Builder (CRITICAL - NOT DONE)
**Required**: `skyarena2d/rl/adapters/policy_obs_builder.py`
- Must build MAPPO-compatible observations from SkyArena state
- Fields needed:
  - self_features (N, 20)
  - entity_features (N, slots, 10) - FILTERED by visible_matrix
  - entity_mask, candidate_ids, candidate_can_long/short
  - semantic_map (N, 9, 100, 100)
  - current_search_goal_id, agent_id, region_features
  - alive_mask, has_active_contact
  - course_mask, search_goal_mask, target_mask
  - global_state (for critic only, unfiltered)
- **CRITICAL**: Must NOT leak invisible enemy positions to policy

### 8. Action Adapter (CRITICAL - NOT DONE)
**Required**: `skyarena2d/rl/adapters/action_adapter.py`
- Convert actor outputs to SkyArena actions
- Handle course/search_goal → heading conversion
- Handle target/fire → hit_targets encoding
- Build action masks

### 9. MAPPO Models (CRITICAL - NOT DONE)
**Required**:
- `skyarena2d/rl/models/actor.py` - Phase1Actor port
- `skyarena2d/rl/models/critic.py` - CentralizedCritic port
- `skyarena2d/rl/models/encoders.py` - EntityEncoder, SemanticMapEncoder, mlp

### 10. MAPPO Algorithm (CRITICAL - NOT DONE)
**Required**:
- `skyarena2d/rl/algo/rollout.py` - GAE, sampling, RolloutBatch
- `skyarena2d/rl/algo/search_goal_manager.py` - TeamSearchPlanner port
- `skyarena2d/rl/algo/mappo_trainer.py` - Main training loop

### 11. Environment Adapter (CRITICAL - NOT DONE)
**Required**: `skyarena2d/rl/adapters/skyarena_mappo_env.py`
- Wraps SkyArenaEngine for MAPPO training
- Handles red (policy) vs blue (fix_rule_v2)
- Returns MAPPO-compatible obs/reward/done/info

### 12. Training Scripts (CRITICAL - NOT DONE)
**Required**:
- `scripts/train_mappo.py`
- `scripts/evaluate_mappo.py`
- `configs/mappo_skyarena.yaml`

### 13. GUI Scoreboard (NOT DONE)
**Required**: Modify `skyarena2d/render/pixel_renderer.py`
- Add scoreboard_height = 80
- Modify _world_to_screen to avoid scoreboard area
- Add _draw_scoreboard() method
- Display metrics in top bar

### 14. Trace Logging (NOT DONE)
**Required**:
- `skyarena2d/logging/trace_recorder.py`
- `skyarena2d/logging/episode_summary.py`
- Modify `scripts/eval_rule_vs_rule.py` to support --record_trace

### 15. Tests (NOT DONE)
**Required**:
- `tests/test_policy_obs_no_enemy_leak.py`
- `tests/test_action_adapter_shapes.py`
- `tests/test_fireable_and_selected_metrics.py`
- `tests/test_reward_modules.py`
- `tests/test_fix_rule_v2.py`
- `tests/test_renderer_scoreboard.py`
- `tests/test_mappo_adapter_shapes.py`

## Implementation Priority

### Phase 1: Core Training Infrastructure (CRITICAL)
1. Policy obs builder (no enemy leak)
2. Action adapter
3. MAPPO models (actor, critic, encoders)
4. Rollout & search_goal_manager
5. MAPPO trainer
6. Environment adapter
7. Training scripts & config

### Phase 2: Validation & Testing
8. All 7 test files
9. Smoke test runs

### Phase 3: Observability
10. GUI scoreboard
11. Trace logging
12. Episode summary

## Key Design Decisions Made

1. **No MaCA API replication**: Using SkyArena-native state/cache
2. **CTDE separation**: Policy obs filtered, critic global_state unfiltered
3. **Modular rewards**: Config-driven enable/disable
4. **Attempted/Selected distinction**: Clear 3-level tracking (fireable/attempted/selected)
5. **Fix rule v2**: Initial push + persistent target tracking
6. **Reward components in info**: Full transparency for debugging

## Risks & Mitigation

### Risk 1: Policy Obs Enemy Leak
**Mitigation**: Test `test_policy_obs_no_enemy_leak.py` must verify invisible enemies have zero features

### Risk 2: Action Adapter Shape Mismatch
**Mitigation**: Test `test_action_adapter_shapes.py` must verify all dimensions

### Risk 3: Reward Double-Counting
**Mitigation**: Terminal win/loss only applied when `termination_result.done == True`

### Risk 4: MAPPO Model Port Errors
**Mitigation**: Start with minimal model, verify shapes before full training

## Next Steps

1. Implement policy_obs_builder.py (highest priority)
2. Implement action_adapter.py
3. Port MAPPO models from MaCA (remove MaCA dependencies)
4. Implement training loop
5. Write tests
6. Run smoke tests
7. Add GUI scoreboard
8. Add trace logging

## Estimated Remaining Work

- **Core training (Phase 1)**: ~2000 lines of code
- **Tests (Phase 2)**: ~800 lines of code
- **Observability (Phase 3)**: ~500 lines of code
- **Total**: ~3300 lines remaining

## Files Modified So Far

1. `skyarena2d/core/state.py` - Extended with tracking fields
2. `skyarena2d/core/weapons.py` - Added attempted/selected tracking
3. `skyarena2d/core/metrics.py` - Added 15+ new metrics
4. `skyarena2d/core/reward.py` - Modular reward system
5. `skyarena2d/core/config.py` - Added reward_modules field
6. `skyarena2d/opponents/fix_rule_v2.py` - New opponent
7. `skyarena2d/opponents/__init__.py` - Registered fix_rule_v2

## Files Created (Empty Directories)

- `skyarena2d/rl/models/`
- `skyarena2d/rl/algo/`
- `skyarena2d/rl/adapters/`
- `skyarena2d/rl/utils/`
- `skyarena2d/core/reward_modules/`
- `skyarena2d/logging/`
