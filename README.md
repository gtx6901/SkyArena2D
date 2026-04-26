# SkyArena2D

## 1. 项目简介

SkyArena2D 是一个透明、可配置、可训练、可渲染的 2D 俯视图多智能体空战环境，用于替代 MaCA 黑盒环境。核心目标：

- 10v10 同构 fighter 默认场景
- fighter/detector 两类作战单元
- 多频点雷达与干扰对抗
- 被动探测与 MaCA-like 原始观测兼容
- 导弹即时/延迟结算与同步结算
- 现代 dense observation
- PettingZoo ParallelEnv 封装
- human GUI 与 rgb_array 渲染

## 2. 安装方式

```bash
cd skyarena2d
/usr/bin/python3 -m pip install -e .
/usr/bin/python3 -m pip install -e .[dev]
```

## 3. 运行 smoke test

```bash
cd skyarena2d
/usr/bin/python3 scripts/smoke_test_env.py --config configs/env_10v10_full.yaml --steps 100
```

## 3.1 训练与评估文档

MAPPO 训练、smoke/train 配置、policy eval 与 GUI eval 课表，请参考：

- `docs/TRAINING.md`

## 4. 运行 rule vs rule eval

```bash
cd skyarena2d
/usr/bin/python3 scripts/eval_rule_vs_rule.py --red rush_rule --blue patrol_rule --episodes 20 --config configs/env_10v10_full.yaml
```

## 5. 打开 GUI

```bash
cd skyarena2d
/usr/bin/python3 scripts/play_gui.py --red rush_rule --blue fix_rule_like --config configs/env_10v10_full.yaml --speed 30
```

GUI 热键：

- `Space`: 暂停/继续
- `N`: 单步推进
- `D`: 切换 debug overlay
- `Esc`: 退出

## 6. 环境配置说明

配置位于 `configs/*.yaml`，主要分块：

- `map`: 地图宽高
- `teams`: 红蓝 fighter/detector 数量
- `spawn`: fixed_scaled / random_edge / symmetric_random / curriculum
- `dynamics`: instant 航向更新与边界模式
- `radar`: 频点数、量程、视场角
- `jamming`: deterministic/probabilistic，spot/barrage
- `passive_detection`: 被动侦收范围与开关
- `weapon`: 长/短导弹参数、延迟、同步、被动开火许可
- `reward`: valid/invalid fire、击杀、损失、回合奖励
- `render`: 分辨率与调试显示开关

支持异构 fighter：

- `red_fighter_profiles`
- `blue_fighter_profiles`

每个 profile 可覆盖 speed/radar_range/radar_fov_deg/jammer_range/long_range/short_range/hit_prob/ammo。

## 7. MaCA-like action/raw obs/reward 说明

### Action

fighter_action: `[num_fighter, 4]`

- `[course, radar_freq, jammer_freq, hit_target]`
- `hit_target`:
  - `0`: no fire
  - `1..N`: long missile target id
  - `N+1..2N`: short missile target id

detector_action: `[num_detector, 2]`

- `[course, radar_freq]`

### Raw Obs

每侧返回：

- `detector_obs_list`
- `fighter_obs_list`
- `joint_obs_dict`

字段覆盖 MaCA-like 兼容要求，包括：

- `r_visible_list`
- `j_recv_list`
- `striking_list`
- `striking_dict_list`
- `last_action`
- `last_reward`

### Reward

训练主 reward 语义：

- `reward["red"] / reward["blue"]`: team-level reward（训练使用）
- `reward["red_unit"] / reward["blue_unit"]`: unit-level reward 向量
- `reward["red_unit_sum"] / reward["blue_unit_sum"]`: 仅用于诊断

`info["maca_reward"]` 提供：

- `side1_detector_reward`
- `side1_fighter_reward`
- `side1_round_reward`
- `side2_detector_reward`
- `side2_fighter_reward`
- `side2_round_reward`

## 8. Modern dense obs 说明

每侧 `obs[side]["modern"]` 返回：

- `self`: `[num_agents, 14]`
- `allies`: `[num_agents, num_agents-1, 10]`
- `enemies`: `[num_agents, num_enemies, 13]`
- `masks`: ally/enemy/self_alive
- `global_state`: `[12]`
- `visible_matrix`
- `fireable_long`
- `fireable_short`

特征包含相对位置、归一化距离、bearing、alive、unit_type、speed、ammo、雷达/干扰状态、visible、fireable。

## 9. PettingZoo ParallelEnv 使用示例

```bash
cd skyarena2d
/usr/bin/python3 - <<'PY'
from skyarena2d.envs.pettingzoo_parallel import SkyArenaParallelEnv

env = SkyArenaParallelEnv("configs/env_10v10_fast.yaml", render_mode="rgb_array")
obs, infos = env.reset(seed=0)

done = False
while not done:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminated, truncated, infos = env.step(actions)
    done = (len(env.agents) == 0) or all(terminated.values()) or all(truncated.values())

env.close()
PY
```

## 10. 当前 MVP 的简化点

- `turn_mode` 目前主实现为 instant（保留 limited_turn_rate 配置位）
- 干扰模型为透明简化版本（spot/barrage + deterministic/probabilistic）
- 被动探测仅输出方向与频点，不做多站定位
- 导弹命中按固定命中率或必中开关
- 渲染为像素风 2D 版本，偏调试可视化

## 11. 后续扩展方向

- 限幅转向与更精细飞行动力学
- 更高保真 ECM/ECCM 建模
- 导弹飞行时间与末制导模型
- 多传感器融合与更复杂被动定位
- 批量环境与并行 rollout 接口
- 训练回放压缩与可视化分析工具

## 12. 如何接入现有 MAPPO 框架

推荐两种接入方式：

1. 直接用 `SkyArenaEngine`：

- 输入 MaCA-like `fighter_action` / `detector_action`
- 读取 `obs[side]["raw"]` 与 `info["maca_reward"]`

1. 用 `SkyArenaParallelEnv`：

- 直接接 PettingZoo-compatible 多智能体训练管线
- 通过 wrapper 将策略动作编码为 MultiDiscrete

最小对接流程：

```bash
cd skyarena2d
/usr/bin/python3 - <<'PY'
import numpy as np
from skyarena2d.core.engine import SkyArenaEngine

env = SkyArenaEngine("configs/env_10v10_fast.yaml")
obs, info = env.reset(seed=0)

for _ in range(10):
    red_f = np.zeros((10, 4), dtype=np.float32)
    red_d = np.zeros((0, 2), dtype=np.float32)
    blue_f = np.zeros((10, 4), dtype=np.float32)
    blue_d = np.zeros((0, 2), dtype=np.float32)
    obs, reward, done, trunc, info = env.step({
        "red": {"fighter_action": red_f, "detector_action": red_d},
        "blue": {"fighter_action": blue_f, "detector_action": blue_d},
    })
    if done:
        break

env.close()
PY
```
