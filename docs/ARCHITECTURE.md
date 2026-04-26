# SkyArena2D 架构说明

## 项目定位

SkyArena2D 是一个透明、可控、可诊断的 2D 多智能体空战环境。

当前状态：

- 环境内核和 rule-vs-rule 流程已经可以支撑日常开发。
- GUI、scoreboard、trace 和 summary 可以用于调试。
- MAPPO 训练管线已经能运行，但仍属于实验层。
- 训练收敛质量尚未作为稳定能力承诺。

## 模块职责

### `core`：稳定环境内核

主目录：[skyarena2d/core](../skyarena2d/core)

职责：

- 状态容器：`EnvState`、`TeamState`、`StepCache`。
- 单步执行：`SkyArenaEngine.step()`。
- 感知和战斗链路：
  - `sensors`
  - `jamming`
  - `passive_detection`
  - `weapons`
  - `metrics`
  - `reward`
  - `termination`

`core` 层应保持和训练算法解耦。环境规则、武器结算、reward 和指标应在这里保持清晰、可测试。

### `training`：策略接口层

主目录：[skyarena2d/training](../skyarena2d/training)

职责：

- 构造 actor 使用的 policy obs。
- 构造 centralized critic 使用的 `global_state`。
- 将 actor 离散输出转换为 `SkyArenaSideAction`。
- 使用 `ew_strategy.py` 中的训练侧电子战启发式，暂时让 jammer 由规则控制。
- 维护 CTDE 边界：
  - actor policy obs 不泄露不可见敌机真值。
  - critic global_state 可以使用全局真值。

这一层是训练算法和环境内核之间的边界。训练主路径应使用 SkyArena-native action 和 obs，不应重新暴露 MaCA 的 `fighter_action` API。

### `rl`：实验性 MAPPO 层

主目录：[skyarena2d/rl](../skyarena2d/rl)

职责：

- actor / critic / encoder。
- rollout 和 GAE。
- MAPPO trainer。
- search goal manager。
- checkpoint 和 TensorBoard 工具。
- MAPPO 环境封装。

当前保证：

- smoke 训练可以跑通。
- 基本 checkpoint、评估和 TensorBoard 逻辑可用。

当前不保证：

- 长训必然收敛。
- self-play 已完成。
- curriculum 已完成。

### `opponents`：规则智能体

主目录：[skyarena2d/opponents](../skyarena2d/opponents)

职责：

- 规则智能体注册表。
- rule-vs-rule 评估基线。
- MAPPO 第一阶段训练中的 blue rule opponent。

`fix_rule_v2` 行为摘要：

- 初始阶段向敌方方向推进，直到首次接敌。
- 接敌后每个 fighter 独立决策。
- 有可见敌机时跟随最近目标，并按可用射程发射。
- 无可见敌机时随机方向搜索，并保持一段时间。
- 电子干扰目前由规则控制，不交给 RL agent 学习。

### `render`：像素渲染和 GUI

主目录：[skyarena2d/render](../skyarena2d/render)

职责：

- 地图像素渲染。
- fighter / detector / radar / jammer / fireable edge / missile 显示。
- 顶部实时 scoreboard。
- `human` 和 `rgb_array` 渲染模式。

scoreboard 用于快速观察：

- step / winner
- alive / kills / losses
- missiles remaining
- fireable edges
- expected exchange
- selected exchange
- fire execution rate
- first contact / first fire opportunity

### `logging`：可观测性

主目录：[skyarena2d/logging](../skyarena2d/logging)

职责：

- step-level JSONL trace。
- episode summary。
- rule-vs-rule 和训练排查所需的结构化输出。

trace 适合排查：

- 可见矩阵
- fireable / attempted / selected
- launch / resolve
- reward components
- metrics snapshot

## 稳定层、实验层和兼容层边界

稳定层：

- `skyarena2d/core`
- rule-vs-rule 评估流程
- rule opponents
- 基础 GUI renderer
- metrics / trace / summary

实验层：

- `skyarena2d/rl`
- MAPPO trainer
- recurrent PPO update
- search goal manager
- semantic-map 分支
- GUI eval 调度
- training-side EW heuristic

兼容层：

- `SkyArenaSideAction.to_maca_fighter_action`
- `build_maca_raw_obs`
- legacy raw obs

兼容层存在是为了让当前 engine 边界平滑过渡，不代表长期目标是复刻 MaCA API。

## 数据流

规则对战：

```text
SkyArenaEngine -> rule action -> engine.step -> metrics / reward / logging / render
```

MAPPO 训练：

```text
SkyArenaMAPPOEnv
  -> TrainingObsBuilder
  -> Actor / Critic
  -> ActionAdapter
  -> SkyArenaSideAction
  -> engine.step
  -> reward / info
  -> rollout / PPO
```

## 关键设计原则

- core 不依赖 rl。
- 训练主路径使用 SkyArena-native 接口。
- actor 不看不可见敌机真值。
- critic 通过 `global_state` 使用集中式真值。
- radar / jammer 暂时先由规则控制，降低 RL 学习变量数量。
- reward 和 metrics 应可解释、可开关、可测试。
