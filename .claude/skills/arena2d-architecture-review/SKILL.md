---
name: arena2d-architecture-review
description: 审查 SkyArena2D 仓库架构、模块边界、依赖方向和可维护性
---

# SkyArena2D 架构审查

## 适用场景

- 新增模块或跨模块改动前评估架构影响
- 审查现有模块边界是否被破坏
- 检查 core/rl/adapters/opponents 之间耦合是否合理

## 检查清单

### 模块边界
- [ ] `skyarena2d/core` 是否保持对 `rl`、`training`、`opponents` 的零依赖
- [ ] `skyarena2d/adapters` 是否只依赖 `core`（不依赖 `rl` 或 `opponents`）
- [ ] `skyarena2d/training` 是否只依赖 `core` 和 `adapters`
- [ ] `skyarena2d/rl` 是否只依赖 `core` 和 `training`（不反向被 core 依赖）
- [ ] `skyarena2d/opponents` 是否只依赖 `core`
- [ ] `skyarena2d/envs` 是否只依赖 `core` 和 `adapters`
- [ ] `skyarena2d/render` 是否只依赖 `core`

### 数据流
- [ ] engine.step() 流程为：decode -> apply_action -> motion -> jamming -> sensors -> weapons -> reward -> termination，新增逻辑是否插入正确位置
- [ ] StepCache 字段是否只在 engine.step() 中写入，不在外部直接修改
- [ ] EnvState 的 rng 是否只通过 engine 内部使用，外部不替换
- [ ] obs/action 转换是否走 adapters 层，不直接在 envs 或 engine 中硬编码

### 配置系统
- [ ] 新配置项是否在 `EnvConfig` 对应子模型中声明（使用 pydantic Field default）
- [ ] YAML 文件是否与 pydantic model 结构一致
- [ ] 是否避免了在 core 模块中硬编码常数（应走 config 或 default）

### 可维护性
- [ ] 新文件是否放在正确的子包中（不在 core 放 RL 逻辑，不在 rl 放对手逻辑）
- [ ] 文件长度是否可控（超过 500 行的应考虑拆分）
- [ ] 函数长度是否可控（超过 80 行的应考虑拆分）
- [ ] 是否存在循环导入

## 输出格式

```
## 架构审查报告

### 审查范围
<列出涉及的文件和模块>

### 边界检查
| 模块对 | 状态 | 说明 |
|--------|------|------|

### 发现的问题
- [严重] ...
- [警告] ...
- [建议] ...

### 结论
通过 / 需要修改后通过 / 不通过
```

## 修改约束

- 不新增不必要的抽象层
- 不为了"平台完整感"引入通用基类或接口
- 如果只是添加一个配置项，不需要创建新模块
- 不修改 core 层接口签名，除非所有调用方已检查

## 注意事项

- `core` 层是稳定层，不应轻易引入破坏性改动
- `reward_modules/` 子目录存在旧版模块化 reward 代码，但当前 `reward.py` 是主入口
- `training` 和 `rl` 存在功能重叠：`training/obs_builder.py` 和 `rl/adapters/` 都涉及 obs 构造，改动时需明确归属
- `adapters/maca_compat.py` 是兼容层，长期可能被移除，不要在其上新增功能
