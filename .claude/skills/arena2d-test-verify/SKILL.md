---
name: arena2d-test-verify
description: 为 SkyArena2D 生成和审查测试，覆盖环境语义、shape/dtype、regression、smoke test
---

# SkyArena2D 测试生成与审查

## 适用场景

- 新增或修改环境功能后需要编写测试
- 审查现有测试覆盖是否足够
- 排查 regression 时指导应该补充哪些测试
- 验证 PR 的质量门槛

## 测试约定

本项目使用标准 `assert` 而非 pytest fixture。测试模式：

```python
def _base_config() -> EnvConfig:
    """构造最小复现场景的 config，覆盖关键参数。"""
    cfg = EnvConfig()
    cfg.teams.red_fighters = 1
    cfg.teams.blue_fighters = 1
    cfg.spawn.jitter = 0.0
    cfg.weapon.attack_effect_delay = 0
    cfg.dynamics.default_fighter_speed = 0.0
    cfg.weapon.hit_prob_enable = False
    return cfg

def test_something() -> None:
    cfg = _base_config()
    env = SkyArenaEngine(cfg)
    env.reset(seed=0)
    state = env.get_state()
    # 手动设置 state 到测试场景
    state.red.pos[0] = np.array([100.0, 100.0], dtype=np.float32)
    state.blue.pos[0] = np.array([150.0, 100.0], dtype=np.float32)
    # 调用 step
    _, reward, _, _, _ = env.step({...})
    # assert 检查
    assert reward["red"] > 0
```

运行：`.venv/bin/python -m pytest tests/test_<name>.py -v`

## 检查清单

### 必须覆盖的测试类别

#### Deterministic Reset
- [ ] `reset(seed=0)` 两次产生相同初始 obs
- [ ] 不同 seed 产生不同 spawn 位置
- [ ] State 字段在 reset 后为非空、shape 正确
- [ ] `missile_queue` 在 reset 后为空

#### Step 输出 shape/dtype
- [ ] `obs["red"]["modern"]["self"]` shape `[N, 14]`
- [ ] `obs["red"]["modern"]["allies"]` shape `[N, N-1, 10]`
- [ ] `obs["red"]["modern"]["enemies"]` shape `[N, M, 13]`
- [ ] `obs["red"]["modern"]["global_state"]` shape `[12]`
- [ ] `reward["red_unit"]` shape `[N]`，dtype float32
- [ ] `info["metrics"]` 包含核心字段且非空
- [ ] `info["maca_reward"]` 包含 6 个字段

#### Reward Regression
- [ ] valid_fire 在目标在射程内时为正
- [ ] invalid_fire（超出射程/无弹药）为负
- [ ] kill 奖励正确发放给 attacker
- [ ] loss 惩罚正确发放给 target
- [ ] 同一 target 的 assist kill 按 contributor 均分
- [ ] win/lose/elimination_bonus 只在 terminal step 触发
- [ ] discovery reward 在首次发现敌人时触发

#### Termination / Truncation
- [ ] 一方全灭 → done=True, winner 正确
- [ ] 双方全灭 → done=True, winner=draw
- [ ] 弹药耗尽且无 pending missile → done=True
- [ ] max_steps → done=True, truncated=True
- [ ] 已 done 的 env 再 step 会抛 RuntimeError

#### Collision / Map Boundary
- [ ] `boundary_mode=clamp` 时 agent 不出界
- [ ] `boundary_mode=bounce` 时 heading 翻转
- [ ] `boundary_mode=kill` 时出界 agent.alive=False

#### Weapons
- [ ] 长程/短程 missile 正确消耗对应 ammo
- [ ] fireable 矩阵考虑距离、ammo、可见性
- [ ] `hit_prob_enable=False` 时 missile 必中
- [ ] `allow_passive_fire=False` 时被动探测不用于开火
- [ ] delay > 0 时 missile 在正确 step 结算

#### Sensors & Visibility
- [ ] 雷达范围内的敌人可见
- [ ] 雷达范围外不可见
- [ ] FOV 边缘情况（omni detector vs directional fighter）
- [ ] jamming 正确遮蔽可见性
- [ ] 阵亡 agent 不可见

#### 训练脚本 Smoke Test
- [ ] `python scripts/train_mappo.py --config configs/mappo_skyarena_smoke.yaml --device cpu` 能启动
- [ ] smoke 训练在 100 步内不 crash

### 不要求覆盖的
- [ ] 全 RL 收敛测试（时间过长）
- [ ] GUI 交互测试
- [ ] 性能 benchmark

## 输出格式

```
## 测试审查报告

### 当前覆盖情况
<按类别列出已有测试文件>

### 缺失覆盖
| 类别 | 缺失项 | 优先级 |
|------|--------|--------|

### 新增测试建议
<具体的测试函数名和测试内容>

### 测试质量
- 是否有 flaky 测试（依赖随机性且 seed 固定的测试）
- 是否有只 assert True 的空测试
```

## 修改约束

- 测试文件放在 `tests/` 目录，命名 `test_<topic>.py`
- 测试函数命名 `test_<description>`，保持 snake_case
- 使用 `from skyarena2d.core.config import EnvConfig` 和 `from skyarena2d.core.engine import SkyArenaEngine` 直接构造环境
- 不要 mock 环境内核（engine、state、weapons）
- 不要依赖外部文件或网络
- 测试应能在 3 秒内完成

## 注意事项

- `test_policy_obs_no_enemy_leak.py` 和 `test_mappo_adapter_shapes.py` 涉及 RL 层，依赖 torch，运行前需确保 torch 可用
- 环境内核测试（weapons、reward、termination、spawn 等）不依赖 torch/Gymnasium/PettingZoo
- `test_fix_rule_v2.py` 使用 rule opponent 的 `act()` 方法，间接测试环境的 obs 结构和对手决策管道
- `test_rollout_target_opportunity.py` 和 `test_fireable_and_selected_metrics.py` 覆盖较新的 fireability 诊断指标
