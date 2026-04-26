# SkyArena2D 训练与评估指南

本文档覆盖环境准备、rule-vs-rule 评估、GUI、MAPPO smoke/train、训练期 GUI eval 课表、参数维护和诊断。

## 1. 环境准备

- Python: `>=3.10`
- 安装依赖:

```bash
python -m pip install -e .
python -m pip install -e .[dev]
python -m pip install -e .[torch]
```

- 验证可导入:

```bash
python -c "import skyarena2d; print('ok')"
```

- 最小 smoke test:

```bash
python scripts/smoke_test_env.py --config configs/env_10v10_full.yaml --steps 100
```

## 2. Rule-vs-rule 评估

```bash
python scripts/eval_rule_vs_rule.py \
  --red fix_rule_v2 \
  --blue fix_rule_v2 \
  --episodes 5 \
  --config configs/env_10v10_full.yaml
```

带 trace 与 summary:

```bash
python scripts/eval_rule_vs_rule.py \
  --red fix_rule_v2 \
  --blue fix_rule_v2 \
  --episodes 3 \
  --config configs/env_10v10_full.yaml \
  --record_trace \
  --trace_dir logs/rule_eval \
  --record_summary \
  --summary_path logs/rule_eval/summary.json
```

## 3. GUI 运行

```bash
python scripts/play_gui.py \
  --red fix_rule_v2 \
  --blue fix_rule_v2 \
  --config configs/env_10v10_full.yaml \
  --speed 30
```

scoreboard 关键字段:

- `step / max_steps`
- `winner / reason`
- `alive`
- `kills / losses`
- `missiles`
- `fireable edges`
- `expected exchange`
- `selected exchange`
- `fire execution rate`
- `first contact / first fire opportunity`

## 4. MAPPO smoke training

```bash
python scripts/train_mappo.py \
  --config configs/mappo_skyarena_smoke.yaml \
  --device cpu
```

## 5. MAPPO 正式训练

```bash
python scripts/train_mappo.py \
  --config configs/mappo_skyarena_train.yaml \
  --device cuda
```

或 CUDA 不可用时:

```bash
python scripts/train_mappo.py \
  --config configs/mappo_skyarena_train.yaml \
  --device cpu
```

说明:

- `configs/mappo_skyarena_smoke.yaml`: 快速联调
- `configs/mappo_skyarena_train.yaml`: 正式训练
- `configs/mappo_skyarena.yaml`: 默认指向训练级配置

## 6. MAPPO 评估

```bash
python scripts/evaluate_mappo.py \
  --config configs/mappo_skyarena_train.yaml \
  --checkpoint path/to/checkpoint.pt \
  --episodes 10 \
  --device cuda
```

## 7. 训练过程中的 GUI eval

默认分两类评估:

- `policy eval`（数值评估）:
  - `evaluation.policy_eval_interval`（默认训练配置 50k）
  - `evaluation.policy_eval_episodes`（默认 5）
- `gui eval`（行为检查）:
  - `evaluation.gui_eval_interval`（默认训练配置 100k）
  - `evaluation.gui_eval_episodes`（默认 1）
  - 默认 `rgb_array` 渲染并保存可视化产物

推荐课表:

- smoke:
  - `save_interval: 2048`
  - `policy_eval_interval: 2048`
  - `gui_eval_interval: 2048`
  - `gui_eval_episodes: 1`
- train:
  - `save_interval: 100000`
  - `policy_eval_interval: 50000`
  - `gui_eval_interval: 100000`
  - `gui_eval_episodes: 1`

CLI 覆盖参数:

```bash
python scripts/train_mappo.py \
  --config configs/mappo_skyarena_smoke.yaml \
  --device cpu \
  --save_interval 2048 \
  --eval_interval 2048 \
  --gui_eval_interval 2048
```

禁用 GUI eval:

```bash
python scripts/train_mappo.py \
  --config configs/mappo_skyarena_train.yaml \
  --disable_gui_eval
```

仅显式开启 human 模式:

```bash
python scripts/train_mappo.py \
  --config configs/mappo_skyarena_smoke.yaml \
  --gui_eval_human
```

注意:

- 默认不启用 human GUI，避免 headless 环境阻塞。
- GUI eval 主要用于行为检查，不宜过于频繁。

## 8. 参数维护指南

### env

- `candidate_slots`: 每个 agent 的候选目标槽位数
- `search_goal_grid_size`: 搜索网格尺寸，网格数为 $G^2$
- `goal_hold_steps`: 搜索目标至少保持步数
- `goal_reach_radius`: 视为到达目标的半径阈值
- `track_memory_steps`: 目标记忆步数
- `blue_rule`: 蓝方规则对手

### train

- `num_envs`: 并行环境数
- `rollout_steps`: 每次 rollout 步数
- `total_env_steps`: 总环境步数
- `save_interval`: checkpoint 间隔
- `eval_interval / policy_eval_interval`: policy eval 间隔
- `gui_eval_interval`: GUI eval 间隔
- `learning_rate`, `gamma`, `gae_lambda`, `clip_coef`, `entropy_coef`, `value_coef`

### reward

- `kill_loss`
- `valid_fire`
- `invalid_fire`
- `fire_execution`
- `selected_exchange`

奖励语义:

- `reward['red'] / reward['blue']`: team-level reward（训练主信号）
- `reward['red_unit'] / reward['blue_unit']`: unit-level reward 向量
- `reward['red_unit_sum'] / reward['blue_unit_sum']`: 仅用于诊断

### render

- `width / height`
- `debug_overlay`
- `show_fireable_edges`
- `show_target_allocations`

## 9. 诊断指标解释

- `red_fireable_edges / blue_fireable_edges`: 当前步可开火边数量
- `expected_exchange_proxy`: 可开火边的期望交换近似
- `selected_expected_exchange`: 实际选中开火边的期望交换
- `fire_execution_rate_given_opportunity`: 有机会时执行开火的比例
- `selected_overkill_mean`: 选中目标的平均过杀度
- `invalid_fire_count`: 非法开火累计计数
- `contact_to_fire_gap`: 首次接触到首次火力机会之间的步差

## 10. 常见问题

- 为什么 actor obs 不能看到不可见敌机？
  - 为避免信息泄漏，actor 只看可见或短时 track-memory 的敌机。

- 为什么 global_state 可以看到全局真值？
  - centralized critic 需要全局状态以稳定训练。

- 为什么训练 reward 使用 team reward，不用 unit reward sum？
  - team reward 避免单位数量线性放大奖励尺度，训练更稳定。

- 为什么 GUI eval 不应太频繁？
  - 渲染与保存会明显拖慢训练吞吐。

- 为什么要分 smoke config 和 train config？
  - smoke 用于快速回归；train 用于长期收敛。

- 如果 CUDA 不可用怎么办？
  - 通过 `--device cpu` 覆盖。

- 如果 headless 环境无法显示 GUI 怎么办？
  - 使用默认 `rgb_array` GUI eval；不要开启 `--gui_eval_human`。

## 补充：reset seed 策略

- 训练默认使用变化 seed:
  - `seed = base_seed + seed_offset * 100000 + reset_counter`
- 评估可使用 deterministic reset:
  - 固定 `base_seed + seed_offset`
- 这样可避免训练时每个 env 每局初始条件重复。
