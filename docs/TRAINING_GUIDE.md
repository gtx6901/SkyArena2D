# SkyArena2D MAPPO 训练指南

## 快速开始

### 1. 环境准备

确保已安装依赖：
```bash
# 激活虚拟环境
source .venv/bin/activate

# 验证 PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 验证 TensorBoard
python -c "from torch.utils.tensorboard import SummaryWriter; print('TensorBoard OK')"
```

### 2. 开始训练

**最小示例（CPU，快速验证）：**
```bash
python scripts/train_mappo.py \
  --config configs/mappo_skyarena.yaml \
  --total_env_steps 1024 \
  --device cpu
```

**完整训练（GPU，推荐）：**
```bash
python scripts/train_mappo.py \
  --config configs/mappo_skyarena.yaml \
  --device cuda
```

训练会自动：
- 创建 `train_dir/skyarena_mappo/` 目录
- 保存 checkpoint 到 `train_dir/skyarena_mappo/checkpoints/`
- 写 TensorBoard 日志到 `train_dir/skyarena_mappo/tb/`

### 3. 监控训练

**启动 TensorBoard：**
```bash
tensorboard --logdir train_dir/skyarena_mappo/tb --port 6006
```

然后在浏览器打开 `http://localhost:6006`

**关键指标：**
- `train/policy_loss` — 策略梯度损失（应逐渐下降）
- `train/value_loss` — 价值函数损失（应逐渐下降）
- `train/entropy` — 策略熵（初始 ~6.93，逐渐下降表示策略收敛）
- `eval/win_rate` — 对 fix_rule_v2 的胜率（目标 >0.5）
- `eval/avg_steps` — 平均回合长度

### 4. 评估训练好的策略

```bash
python scripts/evaluate_mappo.py \
  --config configs/mappo_skyarena.yaml \
  --episodes 20 \
  --device cuda \
  --checkpoint train_dir/skyarena_mappo/checkpoints/step_100000.pt
```

输出示例：
```
[eval] episodes=20 win_rate=0.650 avg_steps=1523.5
```

### 5. GUI 可视化评估

修改 `scripts/play_gui.py` 加载训练好的策略：

```python
# 在 play_gui.py 中添加
from skyarena2d.rl.adapters.skyarena_mappo_env import SkyArenaMAPPOEnv
from skyarena2d.rl.models.actor import SkyArenaActor
import torch

# 加载 checkpoint
checkpoint = torch.load("train_dir/skyarena_mappo/checkpoints/step_100000.pt")
actor = SkyArenaActor(obs_shapes=..., ...)
actor.load_state_dict(checkpoint["actor"])
actor.eval()

# 在主循环中使用 actor 替代 red_rule
```

然后运行：
```bash
python scripts/play_gui.py --config configs/env_10v10_full.yaml
```

---

## 训练配置详解

### configs/mappo_skyarena.yaml

```yaml
train:
  experiment_name: skyarena_mappo_smoke    # 实验名称
  train_dir: train_dir/skyarena_mappo     # 输出目录
  seed: 42                                 # 随机种子
  device: cuda                             # cpu 或 cuda
  num_envs: 4                              # 并行环境数（GPU 推荐 4-8）
  rollout_steps: 128                       # 每次 rollout 采样步数
  total_env_steps: 1000000                 # 总训练步数（1M 约 2-3 小时）
  learning_rate: 0.0003                    # Adam 学习率
  gamma: 0.99                              # 折扣因子
  gae_lambda: 0.95                         # GAE lambda
  clip_coef: 0.2                           # PPO clip 系数
  entropy_coef: 0.01                       # 熵正则化系数
  value_coef: 0.5                          # 价值损失权重
  ppo_epochs: 3                            # 每次 rollout 的 PPO 更新轮数
  max_grad_norm: 0.5                       # 梯度裁剪
  save_interval: 10000                     # checkpoint 保存间隔（步数）
  eval_interval: 10000                     # 评估间隔（步数）
  resume: false                            # 是否从 checkpoint 恢复

env:
  config_path: configs/env_10v10_full.yaml  # 环境配置
  blue_rule: fix_rule_v2                    # 对手规则（当前只支持 fix_rule_v2）
  candidate_slots: 6                        # 观察中的候选目标槽位数
  search_goal_grid_size: 8                  # 搜索目标网格大小（8×8=64 个区域）
  goal_hold_steps: 10                       # 搜索目标保持步数
  goal_reach_radius: 90.0                   # 到达目标的距离阈值

model:
  semantic_map_size: 100                    # 语义地图分辨率
  trunk_dim: 192                            # 主干网络维度
  entity_embed_dim: 96                      # 实体编码维度
  map_embed_dim: 96                         # 地图编码维度
  current_goal_embed_dim: 32                # 当前目标嵌入维度
  agent_id_embed_dim: 16                    # agent ID 嵌入维度
  region_embed_dim: 64                      # 区域特征维度
  lstm_hidden_dim: 192                      # LSTM 隐藏层维度
  critic_hidden_dim: 256                    # Critic 隐藏层维度

logging:
  tensorboard: true                         # 是否启用 TensorBoard
  trace_eval: false                         # 评估时是否记录 trace（调试用）
```

### 关键参数调优建议

**训练速度 vs 样本效率：**
- `num_envs` ↑ → 训练更快，但显存占用更高（GPU 推荐 4-8，CPU 推荐 2-4）
- `rollout_steps` ↑ → 每次更新使用更多样本，但更新频率降低（推荐 64-256）
- `ppo_epochs` ↑ → 每批数据重复使用更多次，但可能过拟合（推荐 3-5）

**探索 vs 利用：**
- `entropy_coef` ↑ → 更多探索，收敛慢但可能找到更好策略（初期 0.01，后期可降到 0.001）
- `clip_coef` ↓ → 更保守的策略更新，更稳定但收敛慢（推荐 0.1-0.3）

**学习率：**
- 初始 `learning_rate: 0.0003` 适合大多数情况
- 如果训练不稳定（loss 震荡），降到 0.0001
- 如果收敛太慢，可升到 0.001（但注意过拟合）

---

## 训练课程表（Training Curriculum）

### 阶段 1：基础对抗（0-100K steps，约 20 分钟）

**目标：** 学会基本的接敌、跟踪、开火

**预期行为：**
- 初始推进阶段能保持队形
- 接触后能锁定最近敌人
- 在射程内尝试开火
- 弹药管理（不浪费）

**评估指标：**
- `eval/win_rate` 达到 0.3-0.4（对 fix_rule_v2）
- `train/entropy` 从 6.93 降到 5.5-6.0
- `metrics/red_fire_execution_rate_given_opportunity` > 0.5

**如果卡住：**
- 检查 `reward_components`：`valid_fire` 和 `kill_loss` 应该有正值
- 降低 `entropy_coef` 到 0.005 加速收敛
- 增加 `num_envs` 到 8 提高样本多样性

### 阶段 2：战术优化（100K-500K steps，约 1.5 小时）

**目标：** 学会协同、目标分配、态势感知

**预期行为：**
- 多个 agent 不重复攻击同一目标（降低 overkill）
- 优先攻击威胁大的敌人
- 利用数量优势形成包围
- 搜索阶段覆盖更多区域

**评估指标：**
- `eval/win_rate` 达到 0.5-0.6
- `metrics/red_selected_overkill_mean` < 0.5
- `metrics/selected_expected_exchange` > 0（正期望交换比）
- `metrics/contact_to_fire_gap` < 10（快速响应）

**如果卡住：**
- 启用 `fire_execution` reward module（奖励有机会时开火）
- 启用 `selected_exchange` reward module（奖励正期望交换）
- 检查 `semantic_map` 是否正确（用 trace logging 可视化）

### 阶段 3：策略精炼（500K-1M steps，约 2 小时）

**目标：** 超越 fix_rule_v2，形成稳定优势

**预期行为：**
- 主动控制交战距离（保持在长程优势区）
- 利用雷达/干扰战术
- 预判敌人移动轨迹
- 损失最小化（避免不必要的交火）

**评估指标：**
- `eval/win_rate` 达到 0.65-0.75
- `metrics/red_losses` < `metrics/blue_losses`（损失更少）
- `train/entropy` 稳定在 4.5-5.5（策略收敛但保留探索）

**如果卡住：**
- 降低 `learning_rate` 到 0.0001（精细调优）
- 增加 `eval_interval` 到 5000（更频繁评估）
- 考虑 curriculum learning（逐步增加敌人数量）

### 阶段 4：泛化测试（1M+ steps）

**目标：** 对不同对手和场景保持鲁棒性

**测试场景：**
```bash
# 对不同规则对手
python scripts/evaluate_mappo.py --episodes 50 --checkpoint step_1000000.pt
# 修改 blue_rule 为 rush_rule, patrol_rule, random_rule

# 不同地图大小
# 修改 configs/env_10v10_full.yaml 中的 map.width/height

# 不同兵力配置
# 修改 teams.red_fighters / teams.blue_fighters
```

---

## 定期评估与可视化

### 每 10K steps 自动评估

训练脚本会自动在 `eval_interval` 时运行评估：
```
[eval] steps=10000 episodes=10 win_rate=0.400 avg_steps=1823.2
[eval] steps=20000 episodes=10 win_rate=0.500 avg_steps=1654.8
[eval] steps=30000 episodes=10 win_rate=0.600 avg_steps=1432.1
```

### 手动带 GUI 评估（推荐每 100K steps）

**方法 1：修改 play_gui.py 加载 checkpoint**

在 `scripts/play_gui.py` 中添加策略加载逻辑（见上文"GUI 可视化评估"）

**方法 2：使用 evaluate_mappo.py + 录制 trace**

```bash
# 评估并录制 trace
python scripts/evaluate_mappo.py \
  --config configs/mappo_skyarena.yaml \
  --episodes 5 \
  --checkpoint train_dir/skyarena_mappo/checkpoints/step_100000.pt \
  --device cuda

# 然后用 eval_rule_vs_rule.py 的 trace 功能可视化
# （需要先实现 trace 到 GUI 的转换工具）
```

**方法 3：TensorBoard 可视化**

TensorBoard 会自动记录：
- 训练曲线（loss, entropy, grad_norm）
- 评估指标（win_rate, avg_steps）
- 奖励分解（reward_components）
- 战术指标（fire_execution_rate, overkill, exchange）

---

## 常见问题

### Q1: 训练不收敛，entropy 一直是 6.93

**原因：** 策略还在随机探索，没有学到有效信号

**解决：**
1. 检查 reward 是否有正值：`info["reward_components"]`
2. 降低 `entropy_coef` 到 0.005
3. 增加 `num_envs` 到 8
4. 确认 `valid_fire` reward module 已启用

### Q2: 训练很快收敛但 win_rate 很低（<0.3）

**原因：** 策略陷入局部最优（例如只学会逃跑）

**解决：**
1. 启用 `fire_execution` reward module（强制开火）
2. 增加 `entropy_coef` 到 0.02（增加探索）
3. 检查 `invalid_fire` penalty 是否过高（降到 -0.01）
4. 重新训练，使用不同 `seed`

### Q3: GPU 显存不足

**解决：**
1. 降低 `num_envs`（4 → 2）
2. 降低 `rollout_steps`（128 → 64）
3. 降低 `semantic_map_size`（100 → 64）
4. 使用混合精度训练（需修改 trainer 代码）

### Q4: 训练速度太慢

**当前性能：** 4 envs × 128 rollout_steps = 512 env_steps/update，约 2-3 秒/update

**优化：**
1. 增加 `num_envs`（4 → 8）
2. 使用更快的 GPU（RTX 4060 → RTX 4090）
3. 减少 `eval_interval`（10000 → 20000）
4. 禁用 TensorBoard（`tensorboard: false`）

### Q5: 如何实现 self-play？

**当前状态：** 只支持 red（policy）vs blue（fix_rule_v2）

**实现 self-play 需要：**
1. 修改 `SkyArenaMAPPOEnv`，blue 也使用 actor
2. 实现 opponent pool（保存历史 checkpoint）
3. 定期从 pool 中采样对手
4. 参考 AlphaStar / OpenAI Five 的 league training

---

## 输出文件结构

```
train_dir/skyarena_mappo/
├── checkpoints/
│   ├── step_10000.pt
│   ├── step_20000.pt
│   └── ...
├── tb/
│   └── events.out.tfevents.xxx
├── config.yaml          # 训练配置备份
└── training.log         # 训练日志（如果启用）
```

**checkpoint 内容：**
```python
{
    "step": 10000,
    "actor": actor.state_dict(),
    "critic": critic.state_dict(),
    "optimizer": optimizer.state_dict(),
    "config": cfg,
}
```

---

## 下一步

1. **完成 100K steps 训练**，验证基础对抗能力
2. **实现 play_gui.py 的策略加载**，可视化评估
3. **调优 reward modules**，提升战术质量
4. **扩展到 self-play**，突破 fix_rule_v2 上限
5. **实现 curriculum learning**，逐步增加难度

---

## 参考资料

- **MAPPO 论文：** [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- **SkyArena2D 环境文档：** `README.md`
- **Reward modules 配置：** `configs/env_10v10_full.yaml`
- **Metrics 说明：** `skyarena2d/core/metrics.py`
