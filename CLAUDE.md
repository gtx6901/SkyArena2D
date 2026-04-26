# SkyArena2D AI Agent 注意事项

## 仓库边界

本项目代码位于 GitHub 仓库：

`gtx6901/SkyArena2D`

默认工作目录应为该仓库根目录。所有代码修改、测试、文档更新都应发生在本仓库内。

不要修改其他仓库。  
不要把旧 MaCA 项目当作需要同步维护的代码库。  
如果需要参考旧 MaCA 代码，只能作为历史参考，不应复制其黑盒接口或强行保持兼容。

## 项目目标

SkyArena2D 的目标是训练出能够击败 `fix_rule_v2` 的红方 RL Agent。

当前训练主线：

- 红方：MAPPO / CTDE / recurrent actor
- 蓝方：rule-based opponent，默认 `fix_rule_v2`
- RL 负责：搜索、机动、目标选择、开火决策
- EW 暂时使用 rule-based heuristic，不纳入 actor action space

除 EW 外，不要继续往红方行为里塞新的手写战术规则。红方主要决策应尽量由 RL policy 学习。

## 工程原则

维护者偏好简洁、清晰、可维护的代码。

请遵守：

- 不写超长文件。
- 不写超长函数。
- 不新增不必要的抽象层。
- 不把多个职责塞进同一个类。
- 不为了“平台完整感”扩展功能。
- 优先小改、可验证、可回滚。
- 不破坏现有 rule eval、GUI eval、MAPPO smoke training。
- 修改前先判断是否直接服务于“击败 `fix_rule_v2`”。

## 训练入口

训练启动应保持为一行命令：

`python scripts/train_mappo.py --config configs/mappo_skyarena_train.yaml --device cuda`

默认行为应写入 YAML，不要要求用户记忆大量 CLI 参数。

## Python 环境

本项目已经有自己的 Python 虚拟环境。

默认不要新建额外虚拟环境，也不要随意切换 Python 解释器。运行脚本、测试和训练前，应优先使用项目已有环境。

如果需要安装依赖，优先使用当前环境执行：

`python -m pip install -e .`

如需开发依赖：

`python -m pip install -e .[dev]`

不要在未确认的情况下执行全局安装，也不要假设系统 Python 就是项目运行环境。

## 文档同步

如果代码修改影响以下内容，必须同步更新文档：

- 启动命令
- 配置项
- 训练流程
- eval / gui_eval 行为
- reward / metrics 语义
- obs / action 语义
- EW 策略
- stable / experimental / legacy 模块边界

优先更新：

- `docs/ENTRYPOINTS.md`
- `docs/TRAINING.md`
- `docs/CONFIG_GUIDE.md`
- `docs/ARCHITECTURE.md`
- `docs/CODE_MAP.md`

不要保留过时状态文档。

## Agent 工作方式

主 Agent 负责：

- 阅读代码；
- 判断问题；
- 设计最小修改方案；
- 审查修改；
- 要求运行必要测试；
- 总结风险。

具体 coding 工作应交给轻量 sub agent。主 Agent 不应一次性重写大块系统，避免污染上下文。

## RL 修改约束

用户不是 RL 专家，因此 RL 相关修改必须保守。

不要随意修改：

- PPO loss
- GAE
- advantage normalization
- recurrent hidden state
- action log-prob 计算
- action mask 语义
- reward scale
- critic input
- actor action space
- rollout storage shape

如果必须修改，先说明：

- 当前问题；
- 为什么现有实现不正确；
- 修改后的训练语义；
- 如何验证；
- 是否影响历史 checkpoint。

## 当前优先级

在扩展新功能前，优先保证：

- red 能发现敌人；
- red 能形成 fireable edges；
- red 能产生 attempted fire；
- red 能产生 selected valid launch；
- red missiles 会正确减少；
- selected_expected_exchange 改善；
- eval win_rate 提升。

不要在这些目标稳定前扩展：

- curriculum
- self-play
- learned EW head
- 新 opponent
- 复杂 reward shaping
- 复杂 EW belief system

## 调试优先级

训练异常时，优先检查：

- `red_fireable_edges`
- `red_attempted_edges`
- `red_selected_edges`
- `red_invalid_fire_count`
- `target_nonzero_rate`
- `fire_nonzero_rate`
- `red_candidate_valid_count`
- `long_available_rate`
- `short_available_rate`
- force-fire 测试

不要只根据 `train/loss` 或 rollout win_rate 判断训练效果。

## Scope Rule

每次修改前先回答：

- 这是否帮助击败 `fix_rule_v2`？
- 这是否增加代码复杂度？
- 是否能用更小的改动完成？
- 是否需要同步更新文档？
- 是否需要新增或更新测试？

如果答案不清楚，先不要改代码。