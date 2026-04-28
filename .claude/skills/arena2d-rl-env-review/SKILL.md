---
name: arena2d-rl-env-review
description: 审查 SkyArena2D 强化学习/仿真环境语义（reset/step/obs/action/reward/termination/info）
---

# SkyArena2D 环境语义审查

## 适用场景

- 修改 `engine.step()` 或 `engine.reset()` 行为
- 修改 obs、action、reward、termination 语义
- 新增或修改 features（如新的 obs 字段、新的 reward component）
- 检查多智能体一致性（red/blue 对称性、agent-level vs team-level）
- 审查训练/评估环境的随机种子与可复现性

## 检查清单

### Reset 语义
- [ ] `reset(seed=N)` 是否产生确定性的初始状态（相同 seed → 相同 spawn、相同 heading）
- [ ] spawn 配置（mode、jitter、spread、heading_spread）在 reset 时被正确使用
- [ ] `rng = np.random.default_rng(seed)` 是否在 reset 中正确接管，引擎内不要使用 numpy global rng
- [ ] reset 返回的 obs 和 info 结构是否与 step 一致
- [ ] episode_idx 是否正确递增
- [ ] reset 时 `missile_queue` 是否被清空（EnvState 默认 factory 已处理）

### Step 语义
- [ ] step 前是否调用了 reset（`self.state is None` 时抛 RuntimeError）
- [ ] 已终止的 episode 是否拒绝继续 step
- [ ] step 执行顺序：decode → apply → motion → jamming → sensors → passive → weapons → reward → termination → cache，不应打乱
- [ ] `state.step_count` 是否在每次 step 递增
- [ ] 双方 `heading` 是否在 step 中被 action 覆写（而非增量叠加）

### Observation
- [ ] `obs["red"]["raw"]` 和 `obs["blue"]["raw"]` 字段与 MaCA-like 文档一致
- [ ] `obs["red"]["modern"]` 包含 `self` (14), `allies` (N-1, 10), `enemies` (M, 13), `masks`, `global_state` (12), `visible_matrix`, `fireable_long`, `fireable_short`
- [ ] actor policy obs 不泄露不可见敌机的真值位置/速度（CTDE 边界）
- [ ] obs shape 在 agent 阵亡后是否正确处理（allies 维度缩减或 mask 设置）
- [ ] `visible_matrix` 与 `fireable_long/short` 在 step 内使用同一步的传感器结果一致

### Action
- [ ] `fighter_action: [num_fighter, 4]` 字段：[course, radar_freq, jammer_freq, hit_target]
- [ ] `detector_action: [num_detector, 2]` 字段：[course, radar_freq]
- [ ] `hit_target` 编码：0=no fire, 1..N=long target, N+1..2N=short target
- [ ] action 超出范围时的 clip 行为是否安全（`decode_maca_side_action` 使用 `% 360.0` 和 `clip`）
- [ ] 阵亡 agent 的 action 是否被忽略（alive mask 在 weapons.py 中检查）

### Reward
- [ ] `reward["red"]` / `reward["blue"]` 是否是 team-level mean reward
- [ ] `reward["red_unit"]` / `reward["blue_unit"]` 是否是 per-unit reward vector
- [ ] reward components 是否可溯源（`info["reward_components"]`）
- [ ] kill/loss 奖励：assist 是否按 contributor 均分
- [ ] valid_fire 奖励和 invalid_fire 惩罚是否互斥
- [ ] win/lose/draw 和 elimination_bonus 是否只在 terminal step 发放
- [ ] discovery reward 是否正确统计了新发现敌人数量

### Termination / Truncation
- [ ] 终止条件：双方全灭、一方全灭、弹药耗尽、max_steps
- [ ] `terminated=True` 时 `truncated` 应为 False（除非 reason=="max_steps" 则 truncated=True）
- [ ] `winner` 字段取值：`"red"` / `"blue"` / `"draw"` / `"ongoing"`
- [ ] 弹药耗尽时 `missile_queue` 为空才触发终止

### Info
- [ ] `info["metrics"]` 字段与 `MetricsTracker` 一致
- [ ] `info["maca_reward"]` 提供 side1/side2 detector/fighter/round reward
- [ ] info 在每个 step 都返回完整结构（不在某些步缺字段）

### 随机种子与可复现性
- [ ] 所有随机性（spawn、jamming、hit_prob）是否通过 `state.rng` 统一控制
- [ ] 给定 seed 后，连续 step 的结果是否确定（给定相同 action sequence）

## 输出格式

```
## 环境语义审查报告

### 审查范围
<涉及的函数/模块>

### 语义一致性检查
| 检查项 | 状态 | 说明 |
|--------|------|------|

### 发现的问题
- [严重] ... (可能导致训练异常)
- [警告] ... (可能导致诊断困难)
- [建议] ...

### 对训练的影响
<如果修改 reward/obs/action，说明对训练 metric 和收敛的潜在影响>
```

## 修改约束

- 不静默修改 reward 语义（scale、sign、发放条件）
- 不修改 action 空间编码方式（除非训练适配层同步更新）
- 不修改 observation 的字段顺序或 shape（历史 checkpoint 依赖）
- 不修改 termination 条件（rule eval 基线依赖当前语义）
- 如果必须修改，先说明问题、正确性、验证方式、对历史 checkpoint 的影响

## 注意事项

- 当前 `hit_prob_enable=True` 时使用 `state.rng.random()` 进行命中判定，需确认 seed 可控
- `attack_effect_delay > 0` 时 missile 结算延迟到未来 step，termination 检查在 delay 期间不考虑 pending missiles
- `allow_passive_fire=False` 是默认值，被动探测信息仅用于观测，不用于开火
- `simultaneous_resolution=True` 时同一 target 的多个 missile 命中概率合并为 `1 - prod(1-p_i)`，当前始终启用
- `reward_modules` 配置支持模块化开关，但 `reward.py` 中的默认行为是传统 reward 全开
