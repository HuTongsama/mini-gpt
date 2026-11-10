# Mini-GPT Stage 3.3 — 完整模型接口契约

**文档版本：** v0.1  
**状态：** 确认  
**前置条件：** Stage 3.2 PASS（累计回归基线：102/102）  
**目标生产文件：** `src/model.py`

## 1. 阶段目标

组装已通过验收的 `InputEmbedding` 和 `TransformerBlock`，得到能够把 Token ID 序列映射为下一 Token 预测分数（Logits）的 **Decoder-only Mini-GPT**。

本阶段只实现**模型结构及前向计算**，不实现训练循环、Cross Entropy Loss、文本生成、Checkpoint 或 Tokenizer 训练。

## 2. 公开接口

```python
import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        ...

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ...
```

### 2.1 参数与形状约束

| 项目 | 契约 |
| --- | --- |
| `vocab_size` | 正整数，词表大小 `V` |
| `context_size` | 正整数，最大上下文长度 `C` |
| `embedding_dim` | 正整数，隐藏维度 `D` |
| `num_layers` | 正整数，Transformer Block 数量 `L` |
| `num_heads` | 正整数，Attention Head 数量 `H` |
| `mlp_hidden_dim` | 正整数，MLP 中间维度 |
| `dropout` | 浮点概率，`0.0 <= dropout < 1.0`；默认 `0.0` |
| `causal` | 布尔值；默认 `True` |
| `token_ids` | `[B, T]` 的 `torch.long` Tensor，`B >= 1`，`1 <= T <= C`，ID 在 `[0, V)` |
| `logits` | `[B, T, V]` 的浮点 Tensor |

要求 `embedding_dim % num_heads == 0`。本阶段基础测试环境为 Windows CPU、`float32`。

以上为新增 **MiniGPT 的公开接口**；内部调用既有模块时，必须适配项目已经验收的真实构造参数名称和行为。不得为了匹配此契约改写 Stage 1～3.2 的公共接口。

## 3. 功能需求

### REQ-3.3.1 — 输入 Embedding

使用已经验收的 `InputEmbedding`，根据 Token ID 和位置产生 `[B, T, D]` 的隐藏状态。

- Batch 内每个样本的位置编号从 0 开始。
- 不直接将 Token ID 作为浮点特征输入 Transformer。
- 不重新实现已存在的 Token Embedding 或 Position Embedding 算法。

### REQ-3.3.2 — 多层 Transformer Block

顺序堆叠 `num_layers` 个已验收的 `TransformerBlock`：

```text
H0 = InputEmbedding(token_ids)
H1 = Block_1(H0)
H2 = Block_2(H1)
...
HL = Block_L(H[L-1])
```

- 各 Block 结构相同，**参数相互独立**，不共享同一个模块实例。
- `embedding_dim`、`num_heads`、`mlp_hidden_dim`、`dropout`、`causal` 配置正确传入。
- 每层输入输出形状均为 `[B, T, D]`。

### REQ-3.3.3 — Final LayerNorm

在最后一个 Transformer Block 后执行独立的、可训练的 `nn.LayerNorm(embedding_dim)`，得到 `[B, T, D]`。

Final LayerNorm 不能替代各 Block 内已有的 LayerNorm。

### REQ-3.3.4 — LM Head

通过独立的线性输出层将隐藏状态映射为词表上的 Logits：

```text
[B, T, D] -> Linear(D, V) -> [B, T, V]
```

- 第一版 LM Head 使用 `nn.Linear(embedding_dim, vocab_size, bias=False)`。
- 第一版**不与输入 Token Embedding 共享权重**，以保持接口与参数关系直观。
- `forward()` 返回原始 Logits，不执行 Softmax、Argmax 或采样。

### REQ-3.3.5 — 因果性与样本隔离

当 `causal=True` 时，位置 `i` 的 Logits 不能依赖位置 `j > i` 的输入。

不同 Batch 样本互不影响。在 `model.eval()` 模式下，同一模型和输入具有确定性。

当 `causal=False` 时，沿用已验收 Block 的双向注意力行为；该模式用于测试和对照，不是 GPT 默认推理方式。

### REQ-3.3.6 — 梯度、模式与错误输入

- 所有可训练模块参与自动微分；合理测试输入下梯度有限。
- `model.train()` / `model.eval()` 正确传递到所有子模块及 Dropout。
- 对无效构造参数明确拒绝（异常类型可采用 Python/PyTorch 的合理类型）。
- 越界 Token ID、`T > context_size` 不应被静默接受。
- 不要求对任意有限极值输入保证无 NaN/Inf；沿用此前约定的代表性数值测试范围。

## 4. 默认教学配置

```text
vocab_size       = 从 Stage 1 Tokenizer 获取
context_size     = 64
embedding_dim    = 128
num_layers       = 2
num_heads        = 4
mlp_hidden_dim   = 512
dropout          = 0.1（训练实验配置）
causal           = True
```

注意构造接口默认 `dropout=0.0`，便于数学正确性测试；实际训练配置可以使用 `0.1`。

## 5. 验收标准

- [ ] 公开构造接口和 `forward()` 符合 §2。
- [ ] 输出形状严格为 `[B, T, V]`，包括 `T=1` 和 `T=context_size`。
- [ ] 固定参数条件下，输入 Embedding、Block 串联、Final LayerNorm、LM Head 的执行顺序可独立验证。
- [ ] `num_layers` 个 Block 为不同实例且权重不共享。
- [ ] LM Head 输出为 Logits，而不是概率或已选择的 Token。
- [ ] `causal=True` 时，改变未来 Token 不影响先前位置 Logits，且对应前缀损失对未来输入的依赖为零（可在嵌入层/中间特征层验证）。
- [ ] Batch 之间无信息串扰。
- [ ] 输入及各主要模块参数具有符合预期的梯度传播。
- [ ] Dropout 在训练/评估模式的行为正确。
- [ ] 合法输入不出现意外 NaN/Inf；无效参数、非法 ID、超长序列得到明确错误。
- [ ] Stage 1～3.2 的 102 个历史回归测试全部通过。

## 6. 范围外事项

本阶段不实现：

- 数据集构造、训练循环、优化器、Loss 计算；
- 自回归 `generate()`、采样、KV Cache；
- 加载预训练参数或 Hugging Face 的完整 GPT 模型；
- CUDA / 混合精度优化；
- 自定义 Tokenizer、新的 Attention 算法；
- Checkpoint 存储、恢复训练。

## 7. 交付物

**开发者：** `src/model.py`。  
**Codex：** `tests/test_model.py`、独立测试执行记录和验收报告。

测试文件先于生产实现编写；Codex 不得创建或修改 `src/model.py`。Stage 3.3 通过后，Stage 3 才视为整体完成，随后进入 Stage 4 训练系统。
