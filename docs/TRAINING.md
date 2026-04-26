# SkyArena2D 训练说明

## 当前状态

- MAPPO 训练管线已经可以运行，但仍处于实验阶段。
- smoke 测试通过只说明流程能跑通，不代表训练语义和收敛质量已经完全可靠。
- 目前更适合先做环境、规则对战、指标和 reward 的稳定性验证，再逐步扩大训练规模。

## 关键诊断指标

训练和评估时优先看这些指标：

- `eval/win_rate`
- `gui_eval/win_rate`
- `rollout/win_rate_finished_episodes`
- `eval/avg_steps`
- `red_fireable_edges` / `blue_fireable_edges`
- `expected_exchange_proxy`
- `selected_expected_exchange`
- `fire_execution_rate_given_opportunity`
- `invalid_fire_count`
- `selected_overkill_mean`

其中：

- `fireable` 表示理论可开火机会。
- `attempted` 表示策略尝试选择的攻击边，包含非法尝试。
- `selected` 表示最终合法执行的发射边。
- `expected_exchange_proxy` 反映理论火力交换优势。
- `selected_expected_exchange` 反映实际选择后的火力交换优势。

### Fire Head 诊断

eval report 新增以下 fire head 诊断指标，用于判断 fire head 塌缩原因：

- `fire_argmax_nonzero_rate` — argmax 选非 no-fire 的比例
- `fire_noop_prob_mean` — softmax 后 no-fire 通道的平均概率
- `fire_long_prob_mean` / `fire_short_prob_mean` — long/short 通道平均概率
- `fire_entropy_mean` — fire 分布的平均熵（接近 1.099 = 均匀分布）
- `fire_valid_mask_long_rate` / `fire_valid_mask_short_rate` — fire mask 中 long/short 合法的比例

判断逻辑：

| 现象 | 可能原因 |
|---|---|
| `fire_action_nonzero_rate` ≈ 0，`fire_argmax_nonzero_rate` ≈ 0，`fire_noop_prob_mean` 很高 | fire head 偏向 no-fire |
| `fire_action_nonzero_rate` ≈ 0，`fire_valid_mask_*_rate` ≈ 0 | 合法开火机会太少，fire mask 封锁 long/short |
| `target_action_nonzero_rate` > 0，`fire_action_nonzero_rate` = 0 | target 已启动但 fire head 未启动 |

### Eval Seed 说明

- deterministic policy eval 不等于每 episode 相同 reset seed
- 同一 eval 内每个 episode 使用 `base_seed + seed_offset + episode_idx` 生成不同初始状态
- episode record 中 `seed` 字段记录本 episode 的实际 seed

### 指标口径说明

eval report 中指标按口径分为两类：

- `*_mean` 后缀：step-mean 口径，在 episode 内每步平均
- 无 `_mean` 后缀的累计指标（如 `red_kills`、`missiles_launched_long`）：episode-final 口径

### GUI 渲染说明

在 GUI debug overlay 中，两种线的含义不同：

- **虚线暗橙线（fireable edge）**：仅表示理论可开火机会（fireable edges），不代表导弹已发射。由 `show_fireable_edges` 控制。
- **实线亮色线 + 端点圆点（missile launch）**：表示实际合法发射的导弹。长程导弹为亮黄色，短程导弹为亮红橙色。由 `show_target_allocations` 控制。

Scoreboard 第二行新增 `attempt`、`selected`、`invalid` 计数（R/B），分别对应红蓝双方的尝试攻击边数、合法发射边数和非法开火次数。

## 训练产物管理

不要把训练产物提交到 git：

- 训练日志
- checkpoint
- GUI eval 输出
- trace JSONL
- 临时评估结果

这些文件应放在 `train_dir/`、`logs/` 或本地临时目录中。

## GUI Eval 策略

GUI eval 的 episode 数可以较多，例如正式训练中保留 `gui_eval_episodes: 15`。这用于获得更稳定的评估指标，不应为了省磁盘而减少评估次数。

默认配置下：

- GUI eval 仍会按 `gui_eval_interval` 运行。
- 仍会记录 TensorBoard 指标，例如 `win_rate`、`episode_len`、`red_kills`、`blue_kills`。
- 默认 `gui_eval_save_video: false` 且 `gui_eval_save_frames: false`。
- 不保存每一步 `frame_XXXXX.npz`，也不生成 mp4/gif。
- 正式训练配置 `configs/mappo_skyarena_train.yaml` 默认使用 `human` GUI eval，跑到 `gui_eval_interval` 时会弹出 live 窗口，便于人工观察训练成果。
- smoke 和通用配置默认使用 `rgb_array`，不会弹出窗口；启动日志会打印 `mode=rgb_array, no window will be opened`。
- 如果某个配置是 `rgb_array`，也可以显式传入 `--gui_eval_human` 临时打开 live GUI。
- `rgb_array` GUI eval 默认每 `gui_eval_render_every: 20` step 渲染一次，用于降低评估阶段的额外开销；`human` GUI eval 会逐步刷新窗口。
- GUI eval 可能较慢，训练器会按 episode 打印开始/结束进度，避免看起来像静默卡住。

如果确实需要视觉回放，再手动开启帧保存；开启前应确认磁盘空间足够。

对 smoke 或通用配置临时打开 live GUI：

```bash
python scripts/train_mappo.py --config configs/mappo_skyarena_smoke.yaml --device cpu --gui_eval_human
```

## Win Rate 口径

`rollout/win_rate_finished_episodes` 和 `eval/win_rate` 不是同一口径：

- `rollout/win_rate_finished_episodes` 只统计训练 rollout 窗口内自然结束的 episode。很多 rollout 可能没有完整结束局，因此这个指标波动大，甚至长期为 0。
- `eval/win_rate` 是固定评估流程下的标准红方胜率。达到 `max_steps` 仍未结束的 episode 会按 draw/truncated 处理，不会沿用上一帧 winner。
- `gui_eval/win_rate` 与 eval 口径一致，但额外经过 renderer 路径，可用于观察策略和 GUI 是否正常。

判断策略效果时优先看 `eval/win_rate`、`eval/avg_return`、`gui_eval/win_rate` 和 GUI 行为，不要把 rollout 胜率直接等同于标准评估胜率。

## 排查训练问题的推荐顺序

1. 先跑 rule-vs-rule，确认环境和规则基线行为正常。
2. 检查 reward scale 是否合理。
3. 检查 policy obs 是否没有泄露不可见敌机真值。
4. 检查 target mask 是否正确。
5. 检查 fire execution 是否长期为 0。
6. 检查 invalid fire 是否过高。
7. 检查 LSTM hidden state 是否在 episode reset 时正确清零。
8. 最后再调 MAPPO 超参数。

## 常用入口

规则对战 sanity check：

```bash
python scripts/eval_rule_vs_rule.py --red fix_rule_v2 --blue fix_rule_v2 --episodes 5 --config configs/env_10v10_full.yaml
```

GUI 可视化：

```bash
python scripts/play_gui.py \
  --red fix_rule_v2 \
  --blue fix_rule_v2 \
  --config configs/env_10v10_full.yaml \
  --speed 30
```

MAPPO smoke：

```bash
python scripts/train_mappo.py --config configs/mappo_skyarena_smoke.yaml --total_env_steps 1024 --device cpu
```

MAPPO 正式训练：

```bash
python scripts/train_mappo.py --config configs/mappo_skyarena_train.yaml --device cuda
```

禁用 GUI eval 的训练 smoke：

```bash
python scripts/train_mappo.py --config configs/mappo_skyarena_smoke.yaml --total_env_steps 1024 --device cpu --disable_gui_eval
```

MAPPO checkpoint 评估：

```bash
python scripts/evaluate_mappo.py --config configs/mappo_skyarena_train.yaml --checkpoint <path> --episodes 10 --device cuda
```
