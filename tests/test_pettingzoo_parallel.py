from __future__ import annotations

from pathlib import Path

from skyarena2d.envs.pettingzoo_parallel import SkyArenaParallelEnv


def test_parallel_env_smoke() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "env_10v10_fast.yaml"
    env = SkyArenaParallelEnv(str(config_path), render_mode="rgb_array")
    obs, infos = env.reset(seed=0)

    assert set(obs.keys()) == set(env.possible_agents)
    assert set(infos.keys()) == set(env.possible_agents)
    assert len(env.agents) == len(env.possible_agents)

    first = env.possible_agents[0]
    assert env.action_space(first) is not None
    assert env.observation_space(first) is not None

    done = False
    steps = 0
    while not done and steps < 1200:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminated, trunc, infos = env.step(actions)
        assert set(rewards.keys()) == set(terminated.keys()) == set(trunc.keys()) == set(infos.keys())
        done = all(terminated.values()) or all(trunc.values()) or len(env.agents) == 0
        steps += 1

    assert done
    env.close()
