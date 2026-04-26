# SkyArena2D 代码地图

本文档用于快速定位“想改某个行为时应该看哪里”。

| 需求 | 应该看哪个文件 | 说明 |
|---|---|---|
| 修改地图尺寸 | `configs/env_10v10_full.yaml` | `map.width` / `map.height` |
| 修改双方数量 | `configs/env_10v10_full.yaml` | `teams.red_fighters` / `teams.blue_fighters` |
| 修改出生位置 | `configs/env_10v10_full.yaml` + `skyarena2d/core/spawn.py` | 先改配置，必要时再改 spawn 逻辑 |
| 修改运动规则 | `configs/env_10v10_full.yaml` + `skyarena2d/core/dynamics.py` | 速度、边界、转向 |
| 修改雷达范围 | `configs/env_10v10_full.yaml` | `radar.fighter_range` |
| 修改电子干扰 | `configs/env_10v10_full.yaml` + `skyarena2d/core/jamming.py` | 干扰范围、模式和命中探测逻辑 |
| 修改电子战训练启发式 | `skyarena2d/training/ew_strategy.py` | 当前 jammer 由启发式控制，不是 actor head |
| 修改被动探测 | `configs/env_10v10_full.yaml` + `skyarena2d/core/passive_detection.py` | 雷达/干扰辐射源的被动发现 |
| 修改导弹射程 | `configs/env_10v10_full.yaml` | `weapon.long_range` / `weapon.short_range` |
| 修改武器结算 | `skyarena2d/core/weapons.py` | fireable / attempted / selected / resolve |
| 修改指标 | `skyarena2d/core/metrics.py` | exchange、overkill、execution rate 等 |
| 修改 reward | `configs/env_10v10_full.yaml` + `skyarena2d/core/reward.py` | 优先改权重和模块开关 |
| 修改 reward module | `skyarena2d/core/reward_modules/` | 模块化 reward 的具体实现 |
| 修改 rule opponent | `skyarena2d/opponents/` | `fix_rule_v2` 等规则智能体 |
| 看训练 obs | `skyarena2d/training/obs_builder.py` | policy obs / global_state |
| 看动作解码 | `skyarena2d/training/action_adapter.py` | actor 输出到 `SkyArenaSideAction` |
| 看 native action 类型 | `skyarena2d/adapters/action_types.py` | `SkyArenaSideAction` |
| 看 MAPPO 环境封装 | `skyarena2d/rl/adapters/skyarena_mappo_env.py` | 训练 red，blue 走 rule |
| 看 MAPPO 模型 | `skyarena2d/rl/models/` | actor / critic / encoder |
| 看 MAPPO 训练 | `skyarena2d/rl/algo/mappo_trainer.py` | 训练主循环，仍是实验层 |
| 看 rollout 工具 | `skyarena2d/rl/algo/rollout.py` | 采样、log_prob、GAE |
| 看 search goal | `skyarena2d/rl/algo/search_goal_manager.py` | 搜索区域持久分配 |
| 看 GUI | `skyarena2d/render/pixel_renderer.py` | 地图渲染、debug overlay、scoreboard |
| 看颜色 | `skyarena2d/render/palette.py` | GUI 调色板 |
| 看 trace | `skyarena2d/logging/trace_recorder.py` | step-level JSONL |
| 看 episode summary | `skyarena2d/logging/episode_summary.py` | episode 级统计汇总 |

## 不建议轻易改动的文件

- `skyarena2d/core/engine.py`
- `skyarena2d/core/weapons.py`
- `skyarena2d/core/reward.py`
- `skyarena2d/training/obs_builder.py`
- `skyarena2d/rl/algo/mappo_trainer.py`

这些文件处在关键路径上，改动前建议先写测试或至少跑 smoke。

## 推荐修改原则

- 能改配置就不要先改代码。
- 能在 adapter 层解决的问题，不要污染 core。
- 训练接口保持 SkyArena-native，不要把 MaCA API 重新带回训练主路径。
- actor 的 policy obs 不能泄露不可见敌机真值。
- critic 的 `global_state` 可以使用全局真值，这是 CTDE 边界。
