"""Test MAPPO adapter shapes and interfaces."""
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from skyarena2d.rl.adapters.skyarena_mappo_env import SkyArenaMAPPOEnv
from skyarena2d.rl.models.actor import SkyArenaActor
from skyarena2d.rl.models.critic import SkyArenaCritic
from skyarena2d.rl.utils.config import load_mappo_config
from skyarena2d.adapters.action_types import SkyArenaSideAction


def test_env_reset_obs_shapes():
    """Test that env reset returns correct obs shapes."""
    cfg = load_mappo_config("configs/mappo_skyarena.yaml")
    env = SkyArenaMAPPOEnv(cfg, seed_offset=0)
    obs = env.reset()

    print("Testing env reset obs shapes...")
    assert "self_features" in obs
    assert "entity_features" in obs
    assert "semantic_map" in obs
    assert "region_features" in obs
    assert "global_state" in obs
    assert "current_search_goal_id" in obs

    N = env.red_fighter_num
    S = env.obs_builder.candidate_slots
    G = env.obs_builder.search_goal_grid_size
    M = env.obs_builder.semantic_map_size

    assert obs["self_features"].shape == (N, 20), f"Got {obs['self_features'].shape}"
    assert obs["entity_features"].shape == (N, S, 10), f"Got {obs['entity_features'].shape}"
    assert obs["semantic_map"].shape == (N, 9, M, M), f"Got {obs['semantic_map'].shape}"
    assert obs["region_features"].shape == (N, G * G, 10), f"Got {obs['region_features'].shape}"
    assert obs["global_state"].shape == (181,), f"Got {obs['global_state'].shape}"
    assert obs["current_search_goal_id"].shape == (N,), f"Got {obs['current_search_goal_id'].shape}"
    print("✓ Env reset obs shapes correct")


def test_env_step_returns_correct_types():
    """Test that env step returns correct types."""
    cfg = load_mappo_config("configs/mappo_skyarena.yaml")
    env = SkyArenaMAPPOEnv(cfg, seed_offset=0)
    obs = env.reset()

    print("Testing env step return types...")
    N = env.red_fighter_num
    dummy_action = SkyArenaSideAction(
        course=np.zeros(N, dtype=np.float32),
        radar_freq=np.ones(N, dtype=np.int32),
        jammer_freq=np.zeros(N, dtype=np.int32),
        fire_type=np.zeros(N, dtype=np.int32),
        target_idx=np.full(N, -1, dtype=np.int32),
    )

    next_obs, reward, done, info = env.step(dummy_action)
    assert isinstance(next_obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    print("✓ Env step return types correct")


def test_actor_forward_shapes():
    """Test actor forward pass shapes."""
    cfg = load_mappo_config("configs/mappo_skyarena.yaml")
    env = SkyArenaMAPPOEnv(cfg, seed_offset=0)
    obs_shapes = env.obs_shapes()

    print("Testing actor forward shapes...")
    actor = SkyArenaActor(
        self_dim=obs_shapes["self_features"][1],
        entity_dim=obs_shapes["entity_features"][2],
        map_channels=obs_shapes["semantic_map"][1],
        candidate_slots=6,
        num_agents=env.red_fighter_num,
        course_bins=16,
        search_goal_bins=64,
        region_feature_dim=obs_shapes["region_features"][2],
        trunk_dim=192,
        lstm_hidden_dim=192,
        entity_embed_dim=96,
        map_embed_dim=96,
        semantic_map_size=100,
    )

    obs = env.reset()
    batch_size = 1
    N = env.red_fighter_num

    # Create dummy batch
    batch = {
        "self_features": torch.as_tensor(obs["self_features"][np.newaxis], dtype=torch.float32),
        "entity_features": torch.as_tensor(obs["entity_features"][np.newaxis], dtype=torch.float32),
        "entity_mask": torch.as_tensor(obs["entity_mask"][np.newaxis], dtype=torch.bool),
        "semantic_map": torch.as_tensor(obs["semantic_map"][np.newaxis], dtype=torch.float32),
        "current_search_goal_id": torch.as_tensor(obs["current_search_goal_id"][np.newaxis], dtype=torch.long),
        "agent_id": torch.as_tensor(obs["agent_id"][np.newaxis], dtype=torch.long),
        "region_features": torch.as_tensor(obs["region_features"][np.newaxis], dtype=torch.float32),
    }

    # Flatten for actor
    flat_batch = {
        "self_features": batch["self_features"].reshape(batch_size * N, -1),
        "entity_features": batch["entity_features"].reshape(batch_size * N, *batch["entity_features"].shape[2:]),
        "entity_mask": batch["entity_mask"].reshape(batch_size * N, -1),
        "semantic_map": batch["semantic_map"].reshape(batch_size * N, *batch["semantic_map"].shape[2:]),
        "current_search_goal_id": batch["current_search_goal_id"].reshape(batch_size * N),
        "agent_id": batch["agent_id"].reshape(batch_size * N),
        "region_features": batch["region_features"].reshape(batch_size * N, *batch["region_features"].shape[2:]),
    }

    h = torch.zeros((batch_size * N, 192), dtype=torch.float32)
    c = torch.zeros((batch_size * N, 192), dtype=torch.float32)

    with torch.no_grad():
        out = actor.step(flat_batch, (h, c))

    assert out["course_logits"].shape == (batch_size * N, 16), f"Got {out['course_logits'].shape}"
    assert out["search_goal_logits"].shape == (batch_size * N, 64), f"Got {out['search_goal_logits'].shape}"
    assert out["target_logits"].shape == (batch_size * N, 7), f"Got {out['target_logits'].shape}"
    assert out["fire_logits"].shape == (batch_size * N, 7, 3), f"Got {out['fire_logits'].shape}"
    print("✓ Actor forward shapes correct")


def test_critic_forward_shape():
    """Test critic forward pass shape."""
    cfg = load_mappo_config("configs/mappo_skyarena.yaml")
    env = SkyArenaMAPPOEnv(cfg, seed_offset=0)

    print("Testing critic forward shape...")
    critic = SkyArenaCritic(global_state_dim=181, hidden_dim=256)

    obs = env.reset()
    global_state = torch.as_tensor(obs["global_state"][np.newaxis], dtype=torch.float32)

    with torch.no_grad():
        value = critic(global_state)

    assert value.shape == (1,), f"Got {value.shape}"
    print("✓ Critic forward shape correct")


if __name__ == "__main__":
    test_env_reset_obs_shapes()
    test_env_step_returns_correct_types()
    test_actor_forward_shapes()
    test_critic_forward_shape()
    print("\n✓ All tests passed!")
