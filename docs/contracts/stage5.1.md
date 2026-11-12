# Mini-GPT Stage 5.1 — Greedy 自回归 Token 生成接口契约

**契约版本：** v0.1  
**状态：** 确认  
**前置条件：** Stage 4.3 PASS（累计 338/338 项测试通过）  
**目标生产文件：** `src/generate.py`

## 1. 阶段目标

基于已通过验收的 `MiniGPT` 前向接口，把一段已有 Token ID 序列作为 Prompt，**逐步预测、选择并追加下一个 Token**，直至生成指定数量的新 Token。

本阶段只实现可复现的 **Greedy Decoding（贪心解码）**：每轮取**最后一个位置** Logits 中分数最高的 Token。不实现温度、随机采样、Top-k、Top-p、EOS 提前停止、KV Cache 或文本交互 CLI。先把“训练时所有位置预测”与“生成时只使用最后一个位置”这一区别验证清楚。

## 2. 公开接口

文件：`src/generate.py`

```python
import torch
from torch import nn


def generate_greedy(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    ...
```

### 2.1 参数契约

| 参数 | 要求 |
| --- | --- |
| `model` | 已创建、已加载所需权重的 Causal MiniGPT（或行为兼容的 `nn.Module`），`forward([B,T]) -> [B,T,V]` |
| `prompt_ids` | `torch.long` Tensor，形状 `[B,T0]`，`B>=1`，`T0>=1`；每一行是一条等长 Prompt |
| `max_new_tokens` | 非负 Python 整数；明确拒绝 `bool`；允许 `0` |
| `context_size` | 正 Python 整数；明确拒绝 `bool`；**与调用方创建 MiniGPT 时的上下文容量保持一致** |
| 返回值 | `torch.long` Tensor，形状 `[B,T0+max_new_tokens]`；包含**原始 Prompt + 所有新生成的 Token** |

- 首版限定 Windows CPU、`float32` 的模型计算和 `torch.long` Token ID。
- `model` 和 `prompt_ids` 的设备由调用方负责保持兼容；此函数不调用 `model.to()`，也不暗中移动 Prompt 到其他设备。
- 训练和生成必须使用**同一个 Tokenizer 的词表与 ID 映射**。加载 Stage 4.3 Checkpoint 时，模型架构和词表匹配由调用方保证。
- 传入模型应按 GPT 用途配置为 `causal=True`。本函数不将双向 Attention 模型转换成因果模型。
- `context_size` 显式传入，是为了**不要求修改已验收的 MiniGPT 公开 API，也不依赖它是否暴露同名属性**。

## 3. 生成过程约束

每轮使用当前 Token 序列的**最后 `context_size` 个 Token**作为输入：

```text
Prompt IDs: [B, T0]
         ↓
model(current_ids[:, -context_size:]) -> logits [B, T_window, V]
         ↓
只选 logits[:, -1, :] -> [B, V]
         ↓
argmax(dim=-1) -> next_id [B, 1]
         ↓
将 next_id 追加到 current_ids 的序列末尾
         ↓
重复 max_new_tokens 次
         ↓
输出 [B, T0 + max_new_tokens]
```

上图只描述行为和数据流；开发者负责独立实现，不要求采用指定的局部变量或语句组织。

## 4. 功能需求

### REQ-5.1.1 — 输入校验与输出契约

- 拒绝非 `torch.Tensor`、错误的 `prompt_ids` dtype、非二维输入、空 Batch 或零长度 Prompt。
- 拒绝 `max_new_tokens < 0`、非法整数类型、`bool`；拒绝 `context_size <= 0`、非法整数类型、`bool`。
- 非零生成时，输入 Prompt 被模型实际使用的窗口内，Token ID 必须满足模型原有的词表范围校验；不得静默调整合法 ID。
- 如果 Prompt 长于上下文容量，**只消费其最后一个窗口**；被裁剪掉的历史前缀不要求在本阶段额外逐项校验，但必须在结果中完整保留。
- 返回结果的 dtype、设备与 `prompt_ids` 一致；原始 Prompt 内容和顺序保持不变，不能原地修改输入。
- `max_new_tokens=0` 时，无需调用模型，直接返回与 Prompt 内容相同但独立的 Tensor。

### REQ-5.1.2 — 使用最后位置的原始 Logits

- 每轮仅根据 `logits[:, -1, :]` 选择下一 Token。
- 在词表维使用 `argmax`；无需 Softmax，因为 Softmax 不改变最大值的位置。
- 若有并列最大值，沿用 PyTorch `argmax` 返回**最小 Token ID** 的约定。
- 不得用第一个位置 Logits、沿时间维求最大值、随机选择或直接取 Prompt 最末 Token 充当预测结果。

### REQ-5.1.3 — 真正的自回归反馈

- 每次生成的 `next_id` 必须追加到序列中，并在**下一轮**作为模型输入上下文的一部分。
- 每个 Batch 样本独立选出自己的下一个 Token；不允许把 Batch 样本拼成一条序列或跨样本传递上下文。
- 严格生成 `max_new_tokens` 轮，不提前终止、不多生成；本阶段无 EOS 规则。
- 不进行 Teacher Forcing，也不使用任何真实的“后续目标 Token”。

### REQ-5.1.4 — 上下文窗口滚动

- 每轮传给模型的序列长度不超过 `context_size`。
- 初始 Prompt 已超长时，也应从尾部窗口开始预测，不报“超长上下文”错误。
- 当新生成的 Token 使序列超过上下文容量时，下一轮只输入最新的 `context_size` 个 Token。
- 返回值必须保留完整 Prompt 和全部生成结果；窗口截断**仅作用于模型每轮读取的输入**。
- 每次窗口输入模型时，现有 `InputEmbedding` 按窗口内位置重新编号；不要求设计绝对位置偏移或 KV Cache。

### REQ-5.1.5 — 推理行为与状态保护

- 执行生成计算时使用评估模式，关闭 Dropout，并禁用 Autograd 图构建。
- 生成返回或模型前向抛出异常后，恢复调用前的**模型根模块 train/eval 状态**；不要求恢复调用方人为设置的混合子模块模式。
- 不修改模型参数、持久 Buffer 或已存在的参数梯度；不调用 `backward()` 或优化器。
- Greedy 生成不调用随机采样，也不应主动消耗 PyTorch CPU RNG；相同模型权重与 Prompt 在确定性 CPU 计算条件下输出一致。
- `max_new_tokens=0` 时也不能改变模型模式、参数或随机数状态。

### REQ-5.1.6 — 错误传播与模型兼容性

- 模型不兼容、设备不匹配、窗口内非法 Token ID 或前向失败时，明确失败，不返回伪造或部分生成结果。
- 使用模型公开的 `forward()` 和张量操作，不依赖特定内部字段（如 `model.blocks`、`model.embedding`），不改动 Stage 1～4.3 的生产接口。
- 不要求为任意 NaN/Inf Logits 添加修复逻辑；已有 MiniGPT 正常 `float32` 前向行为是测试范围。不得偷偷改变非有限值以得到可用预测。

## 5. 便于理解的确定性示例

假设词表是 `{0,1,2}`，初始 Prompt 为 `[0,1]`，一个人为控制的测试模型满足：

- 读到末尾为 `1` 的上下文时，最高分 Token ID 是 `2`；
- 读到末尾为 `2` 的上下文时，最高分 Token ID 是 `0`。

那么执行 `max_new_tokens=2` 应返回：

```text
输入：           [0, 1]
第 1 次生成：    [0, 1, 2]
第 2 次生成：    [0, 1, 2, 0]
输出：           [0, 1, 2, 0]
```

这些数值是用于解释接口的**人为规则**，不代表真实 MiniGPT 的输出。真实模型的预测来自 LM Head 所返回的 Logits。

## 6. 验收标准

- [ ] 公开 API 与参数、输出契约一致；`max_new_tokens=0` 不执行模型前向。
- [ ] 独立固定 Logits 案例确认取**最后位置**、在词表维 `argmax` 以及并列最高分规则。
- [ ] 使用状态可观测的测试模型确认每次预测被反馈到下一轮；Batch 样本相互独立。
- [ ] 上下文容量为 1、普通长度和超长 Prompt 均正确滚动截取，且最终输出保留完整 Prompt。
- [ ] 模型在生成时处于 eval/no-grad；正常返回与异常情况下均恢复原有根模块模式。
- [ ] 不更新模型参数、既有梯度或持久 Buffer；生成结果确定、无随机采样。
- [ ] 拒绝无效形状、dtype、整数参数；模型失败时不静默返回。
- [ ] 至少一次真实 `MiniGPT` 集成：小型因果模型输入合法 Prompt，输出形状、合法 Token ID 范围、确定性符合预期。
- [ ] Stage 1～4.3 的 **338 项历史回归**全部通过。

## 7. 非目标与下一步

本阶段不做：

- Temperature、Categorical Sampling、Top-k/Top-p、随机种子控制、EOS 停止。
- KV Cache、增量 Attention 计算、批量动态长度 Prompt、Padding Mask。
- 训练 CLI、交互文本界面、文本流式输出、Tokenizer 重新训练。
- Checkpoint 文件格式改变、跨设备推理优化、混合精度。

**后续建议：** Stage 5.2 再加入 Temperature 和随机采样，并独立验证概率分布、随机数控制及 Top-k；可另行做 Tokenizer 解码与真实语料生成展示。此安排不预先修改 Stage 5.2 的正式契约。

## 8. 交付与职责

**开发者：** 阅读并确认本契约后，独立编写 `src/generate.py`。  
**Codex：** 测试先行，仅编写 `tests/test_generate.py`（及必要的测试专用辅助文件）；不编写、改写生产实现，不修改旧测试、契约或依赖，不 Commit/Push。

如果新生产接口尚不存在，测试先期失败归类为 `NOT_IMPLEMENTED`；如果契约与现有代码冲突，报告 `CONTRACT_AMBIGUITY` 并停止擅自改造。
