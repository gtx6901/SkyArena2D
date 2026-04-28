---
name: arena2d-refactor
description: 约束 SkyArena2D 重构行为，要求最小 diff、保留接口语义、给出验证命令
---

# SkyArena2D 重构约束

## 适用场景

- 重命名模块、类、函数、变量
- 拆分超长文件或函数
- 提取公共逻辑
- 移动代码到不同模块
- 修改接口签名

## 重构原则

1. **最小 diff 优先**：如果改动不影响功能，应该越小越好。不要顺手改格式、换 import 风格、或重写无关代码。
2. **保留公开接口**：`SkyArenaEngine.reset()` / `step()` / `render()` / `close()` 签名不变，所有外部调用方（scripts、tests、training）不应被迫修改。
3. **不静默修改语义**：reward scale/sign、action 编码、obs 字段名/顺序/归一化范围、物理运动方程、termination 条件、weapon 结算逻辑 都不应在重构中改变。
4. **先验证再提交**：每轮重构后立即运行相关测试，确认无 regression。

## 检查清单

### 重构前
- [ ] 是否已定位所有调用点（grep 确认）
- [ ] 是否已阅读相关测试
- [ ] 是否已确认现有测试能覆盖被改动的代码路径
- [ ] 改动是否直接服务于"击败 fix_rule_v2"（如果不是，考虑是否有必要）
- [ ] 是否能用更小的改动完成同样目标

### 重构中
- [ ] 每次只做一种类型的改动（重命名 OR 拆分 OR 移动，不要混杂）
- [ ] 不顺手修改：
  - import 顺序（除非破坏性冲突）
  - 字符串格式
  - 注释
  - 类型注解风格
  - 变量命名（除非是重构目标本身）
- [ ] 移动代码时不改变逻辑（先移，后改）

### 重构后
- [ ] 运行全部环境内核测试：`.venv/bin/python -m pytest tests/ -v --ignore=tests/test_mappo_adapter_shapes.py --ignore=tests/test_action_adapter_shapes.py --ignore=tests/test_policy_obs_no_enemy_leak.py`
- [ ] 运行 smoke test：`.venv/bin/python scripts/smoke_test_env.py --config configs/env_10v10_full.yaml --steps 100`
- [ ] 如果涉及 `training/` 或 `rl/` 模块，运行 RL 相关测试和 smoke training
- [ ] `git diff --stat` 确认改动行数合理（不应出现几百行无关 diff）

## 关键接口契约

以下接口不应在重构中改变签名或语义：

```python
# engine.py
class SkyArenaEngine:
    def __init__(self, config: EnvConfig | str | Path, render_mode: str | None = None) -> None
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]
    def step(self, actions: dict[str, dict[str, np.ndarray]]) -> tuple[dict, dict, bool, bool, dict]
    def render(self, render_mode: str | None = None) -> np.ndarray | None
    def close(self) -> None
    def get_state(self) -> EnvState
    def get_last_obs(self) -> dict | None
    def get_last_info(self) -> dict | None

# state.py - TeamState 字段名和 dtype 是公开契约
# config.py - EnvConfig 及其子模型字段名是公开契约
# reward.py - compute_rewards() 签名和 RewardOutput 字段
# weapons.py - process_weapons() 签名和 WeaponStepResult 字段
# termination.py - check_termination() 签名和 TerminationResult 字段
# sensors.py - compute_visible_matrix() 签名和 SensorResult 字段
```

## 输出格式

```
## 重构计划

### 目标
<一句话描述>

### 影响范围
| 文件 | 改动类型 | 行数估计 |
|------|----------|----------|

### 调用方检查
<列出所有被影响的调用方>

### 验证命令
```bash
.venv/bin/python -m pytest tests/test_xxx.py -v
.venv/bin/python scripts/smoke_test_env.py --config configs/env_10v10_full.yaml --steps 100
```

### 风险评估
- 是否影响训练 checkpoint 兼容性？
- 是否影响 rule eval 基线？
- 是否需要更新文档？

### 结论
可以执行 / 需要调整方案 / 不应执行
```

## 注意事项

- `state.py` 中的 dataclass 使用 `slots=True`，添加新字段需要更新所有构造点
- `config.py` 使用 pydantic BaseModel，重命名字段会影响所有 YAML 配置和代码中的字段访问
- `step()` 返回 5-tuple：`(obs, reward, done, truncated, info)`，与 Gymnasium API 一致，但内部使用 bool 而非 numpy bool
- `engine.step()` 中 `actions` 格式为 `{"red": {"fighter_action": ..., "detector_action": ...}, "blue": {...}}`，不直接使用 PettingZoo 格式
- `adapters/` 中的 `maca_compat.py` 和 `action_types.py` 是兼容层，可以在确认所有调用方已迁移后移除
