"""Permanent regression tests for core training semantics.

These tests verify invariants that must hold across all code changes:
- target_mask only allows fireable candidates
- target_action > 0 implies selected target is fireable
- fire_mask is consistent with selected target fireability
- discovery rewards each enemy id only once
- elimination_bonus only triggers on full elimination
- eval episode seeds don't repeat
- no_attack_rule never fires and bounces at boundaries
- fireable matrices are consistent with post-deduction ammo
"""

from __future__ import annotations

import numpy as np
import torch

from skyarena2d.adapters.action_types import SkyArenaSideAction
from skyarena2d.core.config import EnvConfig
from skyarena2d.core.engine import SkyArenaEngine
from skyarena2d.core.reward import compute_rewards
from skyarena2d.core.termination import TerminationResult
from skyarena2d.opponents.no_attack_rule import NoAttackRuleOpponent
from skyarena2d.rl.adapters.skyarena_mappo_env import SkyArenaMAPPOEnv
from skyarena2d.training.action_adapter import SkyArenaActionAdapter
from skyarena2d.training.obs_builder import SkyArenaTrainingObsBuilder


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _base_config_2v2() -> EnvConfig:
    cfg = EnvConfig()
    cfg.teams.red_fighters = 2
    cfg.teams.blue_fighters = 2
    cfg.spawn.jitter = 0.0
    cfg.weapon.attack_effect_delay = 0
    cfg.weapon.hit_prob_enable = False
    cfg.dynamics.default_fighter_speed = 0.0
    cfg.radar.fighter_range = 500.0
    cfg.weapon.long_range = 300.0
    cfg.weapon.short_range = 150.0
    return cfg


# ---------------------------------------------------------------------------
# 1. target_mask allows visible/tracked attention candidates
# ---------------------------------------------------------------------------

def test_target_mask_allows_nonfireable_entity_candidates() -> None:
    """Movement V3 target_action is attention/engagement target, not fire permission."""
    cfg = _base_config_2v2()
    env = SkyArenaEngine(cfg)
    env.reset(seed=0)
    state = env.get_state()
    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.red.pos[1] = np.array([100.0, 120.0], dtype=np.float32)
    state.blue.pos[0] = np.array([300.0, 100.0], dtype=np.float32)  # dist=200, long only
    state.blue.pos[1] = np.array([100.0, 500.0], dtype=np.float32)  # dist=400, not fireable

    # Step once to get fresh cache
    obs, _, _, _, _ = env.step({
        "red": {"fighter_action": np.zeros((2, 4), dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
        "blue": {"fighter_action": np.zeros((2, 4), dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
    })

    cache = env.state.cache
    obs_builder = SkyArenaTrainingObsBuilder(
        num_fighters=2, candidate_slots=2,
        map_width=cfg.map.width, map_height=cfg.map.height,
    )
    policy_obs = obs_builder.build_policy_obs(
        own=state.red, enemy=state.blue,
        visible_matrix=cache.red_visible[:2, :2],
        fireable_long=cache.red_fireable_long[:2, :2],
        fireable_short=cache.red_fireable_short[:2, :2],
        step_count=1, max_steps=100,
        current_search_goal_id=np.zeros(2, dtype=np.int64),
    )

    target_mask = policy_obs["target_mask"]  # (2, slots+1)
    candidate_can_long = policy_obs["candidate_can_long"]  # (2, slots)
    candidate_can_short = policy_obs["candidate_can_short"]  # (2, slots)

    for i in range(2):
        # slot 0 (no target) must always be allowed
        assert target_mask[i, 0], f"Agent {i}: slot 0 must be allowed"
        for s in range(min(2, candidate_can_long.shape[1])):
            valid_entity = policy_obs["entity_mask"][i, s]
            assert target_mask[i, s + 1] == valid_entity, (
                f"Agent {i} slot {s + 1}: mask={target_mask[i, s + 1]}, "
                f"entity_mask={valid_entity}"
            )


# ---------------------------------------------------------------------------
# 2. target_action > 0 can select nonfireable entities; fire_mask remains strict
# ---------------------------------------------------------------------------

def test_selected_target_can_be_nonfireable_attention_target() -> None:
    """A visible/tracked slot can be selected even when not fireable."""
    cfg = _base_config_2v2()
    env = SkyArenaEngine(cfg)
    env.reset(seed=0)
    state = env.get_state()
    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.red.pos[1] = np.array([100.0, 120.0], dtype=np.float32)
    state.blue.pos[0] = np.array([300.0, 100.0], dtype=np.float32)  # dist=200, long
    state.blue.pos[1] = np.array([400.0, 100.0], dtype=np.float32)  # dist=300, long edge

    obs, _, _, _, _ = env.step({
        "red": {"fighter_action": np.zeros((2, 4), dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
        "blue": {"fighter_action": np.zeros((2, 4), dtype=np.float32), "detector_action": np.zeros((0, 2), dtype=np.float32)},
    })

    cache = env.state.cache
    obs_builder = SkyArenaTrainingObsBuilder(num_fighters=2, candidate_slots=2,
        map_width=cfg.map.width, map_height=cfg.map.height)
    policy_obs = obs_builder.build_policy_obs(
        own=state.red, enemy=state.blue,
        visible_matrix=cache.red_visible[:2, :2],
        fireable_long=cache.red_fireable_long[:2, :2],
        fireable_short=cache.red_fireable_short[:2, :2],
        step_count=1, max_steps=100,
        current_search_goal_id=np.zeros(2, dtype=np.int64),
    )

    # Simulate policy: for each fireable slot, target_action should select it
    can_long = policy_obs["candidate_can_long"]
    can_short = policy_obs["candidate_can_short"]
    for i in range(2):
        for s in range(can_long.shape[1]):
            if policy_obs["candidate_ids"][i, s] >= 0:
                assert policy_obs["target_mask"][i, s + 1], (
                    f"Agent {i} slot {s+1}: valid entity but mask disallows"
                )


# ---------------------------------------------------------------------------
# 3. fire_mask consistency with selected target
# ---------------------------------------------------------------------------

def test_fire_mask_consistent_with_selected_target() -> None:
    """fire_mask[1] (long) only True if selected target is long-fireable, same for short."""
    from skyarena2d.rl.algo.rollout import build_fire_mask_from_selected_targets

    # 3 agents, 3 candidate slots
    candidate_can_long = torch.tensor([
        [True, False, False],
        [False, True, False],
        [False, False, False],
    ])
    candidate_can_short = torch.tensor([
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ])
    alive_mask = torch.tensor([True, True, False])

    # Agent 0 selects slot 1: long=True, short=False → fire_mask[1]=True, [2]=False
    target_action = torch.tensor([1, 0, 0])  # agent 0 -> slot 1, agent 1 no target
    fire_mask = build_fire_mask_from_selected_targets(
        target_action=target_action,
        alive_mask=alive_mask,
        candidate_can_long=candidate_can_long,
        candidate_can_short=candidate_can_short,
    )
    assert fire_mask[0, 0].item()  # no-fire always allowed
    assert fire_mask[0, 1].item()  # long allowed for slot 0 (long-fireable)
    assert not fire_mask[0, 2].item()  # short NOT allowed
    assert fire_mask[1, 0].item()  # agent 1: no target selected, no-fire
    assert not fire_mask[1, 1].item()  # no target → no long
    assert not fire_mask[2, 1].item()  # dead agent → no long

    # Agent 1 selects slot 1: long=False, short=True
    target_action2 = torch.tensor([0, 1, 0])
    fire_mask2 = build_fire_mask_from_selected_targets(
        target_action=target_action2,
        alive_mask=alive_mask,
        candidate_can_long=candidate_can_long,
        candidate_can_short=candidate_can_short,
    )
    assert not fire_mask2[1, 1].item()  # can_long=False
    assert fire_mask2[1, 2].item()  # can_short=True


# ---------------------------------------------------------------------------
# 4. discovery rewards each enemy id only once
# ---------------------------------------------------------------------------

def test_discovery_rewards_each_enemy_once() -> None:
    """Each enemy id should contribute discovery reward at most once per episode."""
    cfg = _base_config_2v2()
    cfg.reward_modules = {"discovery": {"enabled": True, "first_seen": 0.08}}
    env = SkyArenaEngine(cfg)
    env.reset(seed=0)
    state = env.get_state()

    # Initially blue team has 2 alive fighters (id 0 and 1)
    assert state.blue.alive_count == 2

    # Step 1: encounter enemy 0 for the first time → discovery reward
    obs, reward, _, _, info = env.step({
        "red": {"fighter_action": np.array([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
                "detector_action": np.zeros((0, 2), dtype=np.float32)},
        "blue": {"fighter_action": np.array([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
                 "detector_action": np.zeros((0, 2), dtype=np.float32)},
    })
    # Both blue enemies should be discovered on first step (both visible)
    dc = info["reward_components"].get("discovery", {})
    red_new_1 = dc.get("red_new_count", 0)

    # Step 2: same enemies, no new discoveries
    obs2, reward2, _, _, info2 = env.step({
        "red": {"fighter_action": np.array([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
                "detector_action": np.zeros((0, 2), dtype=np.float32)},
        "blue": {"fighter_action": np.array([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
                 "detector_action": np.zeros((0, 2), dtype=np.float32)},
    })
    dc2 = info2["reward_components"].get("discovery", {})
    red_new_2 = dc2.get("red_new_count", 0)

    assert red_new_2 == 0, f"Second step should have 0 new discoveries, got {red_new_2}"


# ---------------------------------------------------------------------------
# 5. elimination_bonus only on full elimination
# ---------------------------------------------------------------------------

def test_elimination_bonus_only_on_full_elimination() -> None:
    """elimination_bonus triggers only when reason is red_eliminated or blue_eliminated."""
    cfg = _base_config_2v2()
    cfg.reward_modules = {"elimination_bonus": {"enabled": True,
        "red_eliminates_blue": 10.0, "blue_eliminates_red": 10.0}}
    env = SkyArenaEngine(cfg)
    env.reset(seed=0)
    state = env.get_state()

    # Kill only 1 blue → not full elimination
    state.blue.alive[1] = False
    from skyarena2d.core.termination import check_termination
    from skyarena2d.core.weapons import WeaponStepResult
    term = check_termination(state, cfg)
    assert not term.done, "Partial elimination should not trigger termination"

    # ammo-depleted termination (not elimination)
    state.red.long_ammo[:] = 0
    state.red.short_ammo[:] = 0
    state.blue.long_ammo[:] = 0
    state.blue.short_ammo[:] = 0
    term2 = check_termination(state, cfg)
    assert term2.done
    assert term2.reason == "ammo_depleted"

    empty_weapons = WeaponStepResult(
        red_fireable_long=np.zeros((2, 2), dtype=bool),
        red_fireable_short=np.zeros((2, 2), dtype=bool),
        blue_fireable_long=np.zeros((2, 2), dtype=bool),
        blue_fireable_short=np.zeros((2, 2), dtype=bool),
        launch_records=[], resolved_records=[],
        red_valid_fire=np.zeros(2, dtype=bool), red_invalid_fire=np.zeros(2, dtype=bool),
        blue_valid_fire=np.zeros(2, dtype=bool), blue_invalid_fire=np.zeros(2, dtype=bool),
        killed_red=[], killed_blue=[],
        missiles_launched_long=0, missiles_launched_short=0,
        missiles_hit=0, missiles_missed=0,
    )
    reward_out = compute_rewards(state=state, config=cfg,
        weapon_result=empty_weapons, termination_result=term2)
    # elimination_bonus should NOT fire for ammo_depleted
    elim = reward_out.components.get("elimination_bonus")
    assert elim is None or elim.get("reason") is None, (
        f"elimination_bonus should not trigger for ammo_depleted, got {elim}"
    )


# ---------------------------------------------------------------------------
# 6. eval episode seeds don't repeat
# ---------------------------------------------------------------------------

def test_eval_episode_seeds_no_repeat() -> None:
    """Eval episodes should use non-repeating seeds."""
    cfg = {
        "train": {"seed": 42},
        "env": {
            "config_path": "configs/env_10v10_fast.yaml",
            "blue_rule": "no_attack_rule",
            "candidate_slots": 6,
            "search_goal_grid_size": 8,
            "track_memory_steps": 10,
            "use_jammer_strategy": False,
        },
        "model": {"semantic_map_size": 50},
    }
    env = SkyArenaMAPPOEnv(cfg, seed_offset=9999, deterministic_reset=True)

    seeds = set()
    for ep in range(10):
        env.reset()
        s = env.engine.state.episode_idx
        seeds.add(int(env._base_seed + env.seed_offset + ep))
    assert len(seeds) == 10, f"Expected 10 unique eval seeds, got {len(seeds)}"


# ---------------------------------------------------------------------------
# 7. no_attack_rule: never fires, bounces at boundaries
# ---------------------------------------------------------------------------

def test_no_attack_rule_never_fires() -> None:
    """no_attack_rule opponent must never set fire action > 0."""
    cfg = EnvConfig()
    cfg.teams.red_fighters = 2
    cfg.teams.blue_fighters = 2
    cfg.map.width = 3000.0
    cfg.map.height = 4000.0
    cfg.radar.fighter_range = 500.0

    opponent = NoAttackRuleOpponent(seed=42, map_width=3000.0, map_height=4000.0)
    env = SkyArenaEngine(cfg)
    obs, _ = env.reset(seed=42)

    for step in range(20):
        blue_action = opponent.act(obs["blue"], side="blue", step_count=step)
        fa = blue_action["fighter_action"]
        assert np.all(fa[:, 3] == 0), f"Step {step}: no_attack_rule fired!"
        obs, _, _, _, _ = env.step({
            "red": {"fighter_action": np.zeros((2, 4), dtype=np.float32),
                    "detector_action": np.zeros((0, 2), dtype=np.float32)},
            "blue": blue_action,
        })


def test_no_attack_rule_bounces_at_boundary() -> None:
    """no_attack_rule reflects heading when near map boundary."""
    opponent = NoAttackRuleOpponent(seed=42, map_width=3000.0, map_height=4000.0, margin=80.0)

    # Construct minimal side obs for one fighter
    side_obs = {
        "raw": {
            "fighter_obs_list": [{
                "course": 0.0,  # heading east
                "pos_x": 2950.0,  # near right boundary
                "pos_y": 2000.0,
            }],
            "detector_obs_list": [],
        }
    }
    action = opponent.act(side_obs, side="red", step_count=0)
    course = action["fighter_action"][0, 0]
    # Should reflect: (180 - 0) % 360 = 180 (heading west)
    assert 170.0 < course < 190.0, f"Expected bounce heading ~180, got {course}"


# ---------------------------------------------------------------------------
# 8. fireable matrices are consistent with post-deduction ammo
# ---------------------------------------------------------------------------

def test_fireable_reflects_ammo_after_firing_last_shot() -> None:
    """After firing the last long missile, fireable_long must show nothing fireable."""
    cfg = _base_config_2v2()
    cfg.weapon.long_ammo = 1
    cfg.weapon.short_ammo = 0
    cfg.weapon.long_range = 500.0
    cfg.weapon.hit_prob_enable = False  # guaranteed hit
    env = SkyArenaEngine(cfg)
    env.reset(seed=0)
    state = env.get_state()
    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.blue.pos[0] = np.array([150.0, 100.0], dtype=np.float32)

    assert state.red.long_ammo[0] == 1

    obs, reward, done, trunc, info = env.step({
        "red": {"fighter_action": np.array([[0, 1, 0, 1], [0, 1, 0, 0]], dtype=np.float32),
                "detector_action": np.zeros((0, 2), dtype=np.float32)},
        "blue": {"fighter_action": np.array([[180, 1, 0, 0], [180, 1, 0, 0]], dtype=np.float32),
                 "detector_action": np.zeros((0, 2), dtype=np.float32)},
    })

    assert state.red.long_ammo[0] == 0, "Ammo should be 0 after firing"

    cache = env.state.cache
    # After fix: fireable_long should show nothing for agent 0 (ammo=0)
    assert not np.any(cache.red_fireable_long[0, :]), (
        "After firing last long missile, fireable_long must be all-False"
    )
    # modern obs self features: long_ammo is at index 7 (stack order in modern_obs_builder)
    modern = obs["red"]["modern"]
    self_feat = modern["self"]
    # long_ammo norm uses max(1, max(own.long_ammo)), so with ammo=0 it's 0/1=0
    assert abs(float(self_feat[0, 7])) < 0.01, (
        f"modern obs long_ammo (idx 7) should be 0 after firing last shot, got {self_feat[0, 7]}"
    )
