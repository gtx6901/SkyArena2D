# SkyArena2D 训练说明

## 当前状态

- MAPPO 训练管线已经可以运行，但仍处于实验阶段。
- smoke 测试通过只说明流程能跑通，不代表训练语义和收敛质量已经完全可靠。
- 目前更适合先做环境、规则对战、指标和 reward 的稳定性验证，再逐步扩大训练规模。

## 关键诊断指标

训练和评估时优先看这些指标：

- `win_rate`
- `avg_steps`
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

## 训练产物管理

不要把训练产物提交到 git：

- 训练日志
- checkpoint
- GUI eval 输出
- trace JSONL
- 临时评估结果

这些文件应放在 `train_dir/`、`logs/` 或本地临时目录中。

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

MAPPO checkpoint 评估：

```bash
python scripts/evaluate_mappo.py --config configs/mappo_skyarena_train.yaml --checkpoint <path> --episodes 10 --device cuda
```
