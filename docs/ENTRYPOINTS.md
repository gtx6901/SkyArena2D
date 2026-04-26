# SkyArena2D 常用入口

本文档列出日常开发、调试、训练和评估时最常用的命令。

## 规则对战

用于快速验证环境内核、武器逻辑、reward、metrics 和规则智能体是否正常。

```bash
python scripts/eval_rule_vs_rule.py --red fix_rule_v2 --blue fix_rule_v2 --episodes 5 --config configs/env_10v10_full.yaml
```

## GUI

用于观察单位运动、雷达范围、电子干扰范围、可开火边、导弹发射和顶部 scoreboard。

```bash
python scripts/play_gui.py --red fix_rule_v2 --blue fix_rule_v2 --config configs/env_10v10_full.yaml --speed 30
```

## MAPPO Smoke

用于确认训练管线可以端到端跑通。这个命令不用于判断收敛质量。

```bash
python scripts/train_mappo.py --config configs/mappo_skyarena_smoke.yaml --total_env_steps 1024 --device cpu
```

## MAPPO 正式训练

用于较长时间训练。默认假设使用 CUDA。

```bash
python scripts/train_mappo.py --config configs/mappo_skyarena_train.yaml --device cuda
```

默认配置会保留 GUI eval 次数并记录指标，但不会保存逐帧文件。需要完全跳过 GUI eval 时可显式关闭：

```bash
python scripts/train_mappo.py --config configs/mappo_skyarena_smoke.yaml --total_env_steps 1024 --device cpu --disable_gui_eval
```

## MAPPO 评估

用指定 checkpoint 跑评估。

```bash
python scripts/evaluate_mappo.py --config configs/mappo_skyarena_train.yaml --checkpoint <path> --episodes 10 --device cuda
```

## TensorBoard

训练日志和评估指标写入 `<exp_dir>/tb/`。启动 TensorBoard：

```bash
tensorboard --logdir train_dir/skyarena_mappo_train/train_v1.2/tb
```

若有多个实验，可以指向父目录一次加载多个 run：

```bash
tensorboard --logdir train_dir/skyarena_mappo_train
```

常用分组：
- `train/*` — PPO loss、pg_loss、vf_loss、entropy
- `rollout_step/*` — 每步平均动作/mask 诊断
- `metrics_step/*` — info["metrics"] 步均诊断
- `episode_mean/*` — 仅在有 episode 结束时写入的 episode 级指标
- `episode/*` — 最终 info["metrics"] 的 episode 级指标
- `eval/*` — policy eval 聚合指标
- `gui_eval/*` — GUI eval 聚合指标

## Trace 评估

用于生成 step-level JSONL trace 和 episode summary，便于排查指标、发射行为和 reward component。

```bash
python scripts/eval_rule_vs_rule.py --red fix_rule_v2 --blue fix_rule_v2 --episodes 2 --config configs/env_10v10_full.yaml --record_trace --trace_dir logs/rule_eval --record_summary --summary_path logs/rule_eval/summary.json
```

## 推荐使用顺序

1. 先跑 rule-vs-rule，确认环境行为正常。
2. 需要视觉排查时再跑 GUI。
3. 修改训练相关代码后先跑 MAPPO smoke。
4. smoke 正常后再跑正式训练。
5. 使用 `evaluate_mappo.py` 固定 checkpoint 做可复现评估。
