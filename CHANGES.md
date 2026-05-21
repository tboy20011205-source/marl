# HHMARL 自定义 PPO 改造说明

## 总体思路

原项目使用 Ray RLlib 封装的 PPO 算法（`ray.rllib.algorithms.ppo.PPOConfig`），PPO 的具体实现（clip、GAE、梯度更新等）隐藏在库内部不可见。

改造目标：**用自定义 PPO 替换 RLlib，算法逻辑完全可见**，同时保持训练流程、环境接口、评估方式不变。

### 架构对比

```
改造前:
  train_hetero.py → PPOConfig (RLlib 黑盒) → build() → algo.train()
  train_hier.py   → PPOConfig (RLlib 黑盒) → build() → algo.train()

改造后:
  train_hetero.py → MultiAgentPPO (自定义) → algo.train()
  train_hier.py   → MultiAgentPPO (自定义) → algo.train()
```

### 设计原则

1. **算法可见** — PPO-Clip、GAE、多智能体中心化评论家全部用纯 PyTorch 实现
2. **接口兼容** — `train()`、`save()`、`restore()`、`compute_single_action()` 等方法签名与原 RLlib 一致
3. **模型独立** — 神经网络不再继承 `TorchModelV2` / `RecurrentNetwork`，改为纯 `nn.Module`
4. **环境不动** — `env_hetero.py`、`env_hier.py`、`env_base.py` 逻辑完全保留

---

## 文件修改详情

### 新建: `algorithms/ppo.py`（核心算法）

| 组件 | 说明 |
|---|---|
| `SlimFC` | 等价 RLlib 的全连接层，支持 `nn.Tanh` 等类级别的 activation |
| `RolloutBuffer` | 存储 rollout 数据 + GAE advantages/returns |
| `PPOPolicy` | 单策略 PPO，支持 MultiDiscrete 动作空间，负责采样、评估、更新 |
| `MultiAgentPPO` | 多智能体 PPO，管理多个 PPOPolicy，处理 rollout 收集、GAE 计算、中心化评论家 |

**PPO 更新核心逻辑**（`PPOPolicy.update`）：
```
ratio = exp(new_log_prob - old_log_prob)
adv_norm = (advantage - mean) / (std + 1e-8)
L_clip = -min(ratio*adv, clip(ratio, 0.75, 1.25)*adv).mean()
L_value = MSE(value, return)
L_total = L_clip + 0.5*L_value - 0.01*entropy
→ Adam optimizer step with grad_norm clip=0.5
```

### 新建: `models/torch_models_hetero.py`（低层策略模型）

原 `ac_models_hetero.py` 中的模型继承自 `TorchModelV2` / `RecurrentNetwork`（RLlib 基类），新文件改为纯 `nn.Module`：

- `Esc1` / `Esc2` — 逃脱策略网络，`forward()` 返回 `(logits, [])`，`value_function()` 返回价值
- `Fight1` / `Fight2` — 战斗策略网络，使用 `MultiheadAttention` 处理时序信息，接口同 Escape

关键改动：去掉 `@override` 装饰器、`SlimFC` 换成本地版本、`add_time_dimension` 换成本地版本。

### 新建: `models/torch_models_hier.py`（高层指挥官模型）

原 `ac_models_hier.py` 的 `CommanderGru` 改为纯 `nn.Module`，GRU 状态管理原样保留。

### 修改: `train_hetero.py`

| 改动点 | 原代码 | 新代码 |
|---|---|---|
| PPO 导入 | `from ray.rllib.algorithms.ppo import PPOConfig` | `from algorithms.ppo import MultiAgentPPO, PPOPolicy` |
| 模型注册 | `ModelCatalog.register_custom_model(...)` | 直接实例化 `Fight1()` / `Esc1()` 等 |
| 算法构建 | `PPOConfig().rollouts(...).training(...).build()` | `MultiAgentPPO(policies, env_fn, ...)` |
| Callback | `CustomCallback.on_postprocess_trajectory` | `postprocess_trajectory()` 函数 |
| TensorBoard | `program.TensorBoard()` 自动启动 | `SummaryWriter`（可选） |

**postprocess 处理要点**：
- 恢复 episode 中所有 agent 的实际动作到增强观测中
- 对 heading 动作除以 12、speed 动作除以 8 做归一化（与原 RLlib callback 一致）
- 对于 `act_2`（友方动作），保留原数组形状，仅覆盖前 3 个元素

### 修改: `train_hier.py`

同 `train_hetero.py` 的改造模式，额外处理：

- **共享策略**：3 个 agent 共用同一个 `CommanderGru` 和 `PPOPolicy`
- **Reward 分配**：通过 `agent_id` 精准匹配每条 transition 的 reward
- **Action 归一化**：Commander 动作除以 `N_OPP_HL`

### 修改: `evaluation.py`

| 改动点 | 原代码 | 新代码 |
|---|---|---|
| 策略加载 | `Policy.from_checkpoint(check, [...])` | `torch.load()` + `SimplePolicy` 包装器 |
| 模型依赖 | 需 `ModelCatalog.register_custom_model` | 直接创建 `CommanderGru()` |

### 修改: `policy_export.py`

| 改动点 | 原代码 | 新代码 |
|---|---|---|
| 模型注册 | `ModelCatalog.register_custom_model(...)` | 直接创建模型实例 |
| 策略导出 | `Policy.from_checkpoint().export_model()` | `torch.load(checkpoint).load_state_dict()` |

### 修改: `envs/env_base.py`

唯一改动：将基类从 `ray.rllib.env.multi_agent_env.MultiAgentEnv` 换为 `gymnasium.Env`，消除项目对 Ray 的最后一项依赖。环境的所有核心逻辑（仿真器管理、状态计算、奖励计算、渲染）不受影响。

### 不修改的文件

- `config.py` — 纯参数解析，不依赖 RLlib
- `envs/env_hetero.py` — 低层环境，接口不变
- `envs/env_hier.py` — 高层环境，接口不变
- `warsim/` — 仿真器，完全不依赖 RLlib
- `models/ac_models_hetero.py` — 原 RLlib 版本保留但不导入
- `models/ac_models_hier.py` — 原 RLlib 版本保留但不导入
