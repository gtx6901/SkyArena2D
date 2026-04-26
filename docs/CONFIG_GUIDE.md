# SkyArena2D 配置说明

## 1. `configs/env_10v10_full.yaml`

这是环境和战斗语义的主配置文件。

主要配置段：

- `map`：地图宽高。
- `teams`：红蓝双方 fighter / detector 数量。
- `dynamics`：速度、转向和边界行为。
- `spawn`：出生位置和随机扰动。
- `radar`：雷达开关、频点数量、探测距离和视场角。
- `jamming`：电子干扰范围和干扰模式。
- `passive_detection`：被动探测规则。
- `weapon`：长/短程导弹射程、命中率、弹药和延迟结算。
- `reward` / `reward_modules`：reward 权重和模块开关。
- `render`：GUI 尺寸、debug overlay、雷达/干扰/可开火边显示开关。

修改环境语义时优先改这个文件。只有配置表达不了的行为，才考虑改代码。

## 2. `configs/mappo_skyarena_smoke.yaml`

用途：

- 验证 MAPPO pipeline 能否端到端跑通。
- 适合 CPU 上快速检查。
- 适合在改动训练接口、obs、action adapter 后第一时间运行。

典型特征：

- `total_env_steps` 小。
- `num_envs` 小。
- `rollout_steps` 小。
- eval 和 save 间隔可以较短。

## 3. `configs/mappo_skyarena_train.yaml`

用途：

- 正式训练或较长实验。
- 通常目标设备是 CUDA。

典型特征：

- `num_envs` 更大。
- `rollout_steps` 更稳定。
- `save_interval` 和 `eval_interval` 不应过短，至少应接近一个 episode 的尺度。
- 不建议频繁改变 reward、环境语义和模型结构。

## 4. `configs/mappo_skyarena.yaml`

这是通用入口脚本默认使用的配置。当前更像一个默认别名配置。

如果要做严肃实验，建议明确使用：

- smoke：`configs/mappo_skyarena_smoke.yaml`
- train：`configs/mappo_skyarena_train.yaml`

## 5. 参数维护建议

- 初期不要频繁改 reward term。
- 先让 rule-vs-rule 的指标合理，再开始调 MAPPO。
- 不要同时修改环境语义、reward 语义和模型结构。
- 每次大改后先跑 smoke，再跑短训练，再跑长训练。
- 对训练结果做对比时，固定 config、seed、checkpoint 和评估 episode 数。

## 6. 配置之间的关系

- `env_10v10_full.yaml`：环境主语义。
- `mappo_skyarena_smoke.yaml`：训练管线 smoke baseline。
- `mappo_skyarena_train.yaml`：正式训练 baseline。
- `mappo_skyarena.yaml`：通用默认配置，适合脚本默认值和快速入口。
