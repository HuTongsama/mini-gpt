# Mini-GPT Stage 5.2 — Temperature、随机采样与 Top-k 接口契约

**契约版本：** v0.1  
**状态：** 确认  
**前置条件：** Stage 5.1 PASS（累计 375/375 项测试通过）  
**目标生产文件：** `src/generate.py`

## 1. 阶段目标与设计边界

Stage 5.1 已实现 `generate_greedy()`：每轮取最后位置 Logits 的最大值，确定性追加下一个 Token。Stage 5.2 在**不改变现有 Greedy 接口和行为**的前提下，增加概率采样路径：

1. **Temperature：** 调整预测分布的尖锐程度。
2. **Categorical Sampling：** 根据分布随机选出一个 Token，而不是恒取最大 Logit。
3. **Top-k：** 只允许当前 Logits 最高的 k 个候选参与采样。
4. **显式随机数生成器：** 支持调用方控制采样的复现性。

本阶段仍然输出 **Token ID**，不实现 Top-p、EOS 提前停止、文本 CLI、KV Cache 或 Tokenizer 改造。Windows CPU、模型 `float32`、Token `torch.long` 为验收范围。

## 2. 公开 API

文件：`src/generate.py`

```python
import torch
from torch import nn


def generate_sample(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    ...
```

### 2.1 参数与输出

| 参数 | 行为契约 |
| --- | --- |
| `model` | 已创建、权重已准备的 Causal MiniGPT 或兼容 `nn.Module`，`forward([B,T]) -> [B,T,V]` |
| `prompt_ids` | `torch.long`、二维 `[B,T0]`、`B>=1`、`T0>=1`；所有 Batch 样本长度一致 |
| `max_new_tokens` | 非负 Python `int`，不接受 `bool`，允许 `0` |
| `context_size` | 正 Python `int`，不接受 `bool`；由调用方保持与模型上下文容量一致 |
| `temperature` | 有限、严格大于 `0` 的实数，不接受 `bool`；支持 `int` / `float` / 合法 `Fraction` 数值，计算时可转换为 `float`；默认 `1.0` |
| `top_k` | `None`（不筛选）或正 Python `int`，不接受 `bool`；真实前向后还必须满足 `top_k <= V` |
| `generator` | `None` 或与 CPU 采样兼容的 `torch.Generator`；不接受其他类型 |
| 返回值 | `torch.long`、`[B,T0+max_new_tokens]`；保留完整 Prompt 及新采样的所有 Token，设备与输入一致 |

- 不调用 `model.to()`，不自动将输入或 Generator 迁移到其他设备。首版只要求 CPU 路径。
- 不改变 `generate_greedy()` 的签名、语义或历史测试。
- 采样不保证两次自然产生相同内容；**相同模型/Prompt、同一随机数初始状态及相同配置应可复现**。
- 训练、Checkpoint 和生成所用 Tokenizer 词表及 ID 映射必须由调用方保持一致。

## 3. 数学语义

设当前最后位置的原始 Logits 为 `z = logits[:, -1, :]`，每行大小为 `V`。

### 3.1 Temperature

对每个候选 Token `i`：

```text
scaled_i = z_i / temperature
```

- `temperature = 1`：保持原始 Logits 的相对尺度。
- `0 < temperature < 1`：分布通常更集中于高分候选。
- `temperature > 1`：分布通常更平缓。
- 任意正温度都不改变有限 Logits 的排序；温度不是 Greedy/随机路径开关。
- `temperature = 0` 非法；Greedy 行为由已有 `generate_greedy()` 提供。

### 3.2 Top-k 候选集

若 `top_k = k`，在**本轮最后位置**的 Logits 中保留分数最高的 `k` 个 Token，其余候选概率严格为 `0`。筛选后的分布必须重新归一化。

- 由于正温度不改变排序，在数值正常时按原始 Logits 或按缩放后 Logits 选前 k 个在数学上等价。
- `top_k=None`：保留全部 V 个候选；`top_k=V` 的概率分布应等价于不过滤。
- 对 **Top-k 截止边界完全相同的 Logits**，本阶段不规定哪一个并列 Token 被保留；独立精确概率测试应使用边界无并列的构造数据。
- `top_k=1` 且最大 Logit 唯一时，无论 Temperature 如何，唯一候选的概率为 1，应与 Greedy 选择一致。

### 3.3 Softmax 与采样

只针对保留的候选，用稳定的归一化获得分类概率：

```text
p_i = exp(scaled_i - m) / sum(exp(scaled_j - m))
m = max(scaled_j)（j 属于保留候选）
```

被 Top-k 排除的候选 `p_i=0`。每行概率均应非负、有限，和为 1（允许合理的 CPU float32 容差）。

每轮独立地对 **每个 Batch 样本**按其概率分布抽取一枚 Token，追加后再进行下一轮模型前向。必须执行**概率采样**，不能将 `top_k>1` 当成最大值选择。

如果模型返回非有限 Logits，或有限 Logits 经极端温度缩放后产生不可用概率，允许以明确异常拒绝；不得静默替换 NaN、Inf 或强行改成 Greedy。本阶段不要求支持任意极值的稳定采样。

### 3.4 可手算参考案例

一行 Logits：

```text
Token ID:        0      1      2
Logits:          0     ln2    ln4
Temperature:     1.0
```

未过滤时概率严格为：

```text
p = [1/7, 2/7, 4/7]
```

若 `top_k=2`，候选为 Token `1` 与 `2`：

```text
p = [0, 1/3, 2/3]
```

若不启用 Top-k，Temperature 改为 `2.0`，相对权重变成 `[1, sqrt(2), 2]`，再除以三者之和。这些是**理论分布参考**，不是要求随机采样必须返回某个指定 Token。

## 4. 功能需求

### REQ-5.2.1 — API、输入校验与零步行为

- `prompt_ids`、`max_new_tokens`、`context_size`、输出独立副本等通用规则与 Stage 5.1 一致。
- 严格拒绝非法 Temperature（布尔、非实数、非有限、零或负值），非法 `top_k`（布尔、非整数、非正值），非法 `generator` 类型。
- 若 `max_new_tokens=0`，完成静态参数校验后**直接返回 Prompt 的独立副本**；不得调用模型、切换模式、采样、修改 Generator 状态或 CPU RNG；无需为了检查 `top_k<=V` 而执行模型前向。
- `top_k>V` 在实际首次获得词表维数后明确失败，不得自动截断为 `V`。

### REQ-5.2.2 — Temperature 分布

- 只读取 `logits[:, -1, :]`，按 `z / temperature` 定义概率分布。
- 使用有数值稳定性意识的 Softmax 或等价方法；不要求对任意溢出尺度提供结果。
- `temperature=1.0` 与原始 Logits 的 Softmax 等价；更小正温度使非并列最高分的相对概率提高，更大正温度使分布更平缓。
- 不自行修改原模型输出的 Logits，也不使用其他时间位置的预测来替代最后位置。

### REQ-5.2.3 — Top-k 精确过滤

- 每轮每个 Batch 样本独立确定 Top-k 候选；其他候选的采样概率严格为 0。
- `None`、`1`、中间整数和 `V` 都要正确；合法候选在过滤后重新归一化。
- `top_k=1`、唯一最大值时结果必须确定；`top_k>1` 仍需真实随机采样。
- 不实现 Top-p、重复惩罚、频率惩罚或自定义 Token 黑名单。

### REQ-5.2.4 — 自回归、Batch 与窗口

- 沿用 Stage 5.1 的尾部 `context_size` 窗口：不足时读取全部，超长时仅喂入尾部。
- 每轮使用采样出的 Token 更新后续上下文，严格生成 `max_new_tokens` 个 Token。
- Batch 内每个样本有独立的概率分布与采样结果；不能跨样本串扰。
- 最终输出必须保留输入 Prompt 的全部 Token（包括被模型读取窗口裁掉的历史前缀）。

### REQ-5.2.5 — 可复现 RNG 与推理状态

- `generator` 提供时，采样随机数必须从该 Generator 获取；在不主动产生随机数的兼容测试模型下，不得额外消耗**全局 CPU RNG**。
- 若使用相同状态初始化的 Generator、相同模型权重、Prompt 和配置，采样输出应该一致；不要求不同实现方式消耗完全相同数量的随机数。
- `generator=None` 时，允许使用 PyTorch 默认 CPU RNG；不强制不同独立调用输出不同。
- 生成前向处于 `eval` 且不构建梯度图；正常返回或抛异常均恢复调用前的**模型根模块**模式。
- 不主动修改模型参数、持久 Buffer 或已有梯度；不要求恢复人为设定的混合子模块模式。
- 零步路径不改变模型、已有梯度、Generator 或 CPU RNG。

### REQ-5.2.6 — 错误传播与历史兼容

- 设备不匹配、模型前向失败、输出结构不兼容、Token ID 越界（已输入窗口内）应明确失败。
- 不用内部实现细节（例如 `model.blocks`）获取输出或上下文长度；仅依赖公开 `forward()` 与传入参数。
- 未定义范围的非有限数值不修补成貌似合法的 Token；不静默返回部分结果。
- `generate_greedy()` 与 Stage 1～5.1 既有生产行为、测试和接口必须保持不变。

## 5. 验收标准

- [ ] API 精确匹配，合法数值参数能参与真实采样；零步不改变任何可观测状态。
- [ ] 使用独立数值参考验证最后位置、Temperature、Softmax、Top-k 后的概率分布；不能仅验证输出形状。
- [ ] 使用测试内替身采样器捕获**实际传入的概率 Tensor**，核对 `[1/7,2/7,4/7]` 和 Top-k 的 `[0,1/3,2/3]` 等精确案例。
- [ ] 非并列 `top_k=1` 与 Greedy 一致；`top_k>1` 能选出非最高分且合法的 Token；被过滤 Token 永远不被选择。
- [ ] 显式 Generator 的复现性、全局 CPU RNG 隔离、零步与异常状态恢复均通过。
- [ ] 自回归反馈、Batch 隔离与完整 Prompt 保留通过；尾部窗口读取与 Stage 5.1 一致。
- [ ] 至少一次真实 `MiniGPT` CPU 小模型集成，生成固定数量的合法 Token，且不修改权重/已有梯度。
- [ ] Stage 1～5.1 的 **375 项历史回归**全部通过。

## 6. 非目标和下一步

本阶段不做：

- Top-p / Nucleus、Beam Search、EOS 终止、Repetition Penalty、动态长度 Batch、Padding Mask。
- KV Cache、流式 Token 输出、CLI、训练语料管理和 Tokenizer 重训练。
- CUDA、混合精度、跨版本采样轨迹完全一致、极端非有限 Logits 自动修复。

**后续可选 Stage 5.3：** 将已实现的 Tokenizer、Checkpoint、Greedy/采样生成接口接成可运行的文本演示，展示训练前后输出差异。是否进入该阶段由开发者在 Stage 5.2 验收后决定。

## 7. 交付与职责

**开发者：** 审阅并确认契约后，亲自修改 `src/generate.py`，独立实现 `generate_sample()`；不提前引入其他采样算法。

**Codex：** 契约测试先行，仅新增/扩展 `tests/test_generate_sample.py` 及必要的测试辅助代码；不修改生产代码、历史断言、既有契约或依赖；不 Commit/Push。

若生产接口尚不存在，分类为 `NOT_IMPLEMENTED`；若与既有契约冲突，报告 `CONTRACT_AMBIGUITY` 而非自行改变生产接口。
