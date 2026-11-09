# Mini-GPT — Stage 3.2 Transformer Block 接口契约

**文档版本：** v0.1  
**状态：** 确认  
**前置阶段：** Stage 3.1 MLP — PASS（本次报告 76/76）  
**目标生产文件：** `src/transformer.py`

## 1. 阶段目标

在已经完成的 `MultiHeadAttention` 与 `MLP` 之上组装一个 **Pre-LN Decoder-only Transformer Block**。本阶段重点理解 LayerNorm、残差连接，以及 Attention 与 MLP 的先后关系。

本阶段只实现一个 Block，不组装完整 GPT、Token Embedding、LM Head、训练循环或 KV Cache。

## 2. 对外接口（待确认后冻结）

```python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

### 2.1 参数契约

| 参数 | 含义 | 约束 |
| --- | --- | --- |
| `d_model` | 输入与输出隐藏维度 | 正整数 |
| `num_heads` | 注意力头数 | 正整数，且 `d_model % num_heads == 0` |
| `mlp_hidden_dim` | MLP 中间层维度 | 正整数 |
| `dropout` | 传递给 MLP 的 Dropout 概率 | 按已验收 Stage 3.1 的公开约定；测试使用 `0.0` 或 `0.1` 等浮点值 |
| `causal` | 是否启用因果注意力 | 默认 `True` |
| `x` | 当前隐藏状态 | `[B, T, d_model]`，`B >= 1`、`T >= 1` |
| 返回值 | 下一层隐藏状态 | `[B, T, d_model]` |

本阶段基础运行环境使用 CPU、`float32`。`T` 的最大长度不由本模块规定；如存在上限，由整体 GPT 的输入/上下文约束处理。

### 2.2 公开子模块

为方便独立测试和检查，模块应公开以下属性：

- `ln1`：`nn.LayerNorm(d_model)`，位于 Attention 前。
- `attn`：现有 `MultiHeadAttention`，`d_model`、`num_heads`、`causal` 按构造参数传递。
- `ln2`：另一个独立的 `nn.LayerNorm(d_model)`，位于 MLP 前。
- `mlp`：现有 Stage 3.1 `MLP`，输入/输出维度为 `d_model`，中间维度为 `mlp_hidden_dim`，使用约定的 `dropout`。

`ln1` 与 `ln2` 具有**互不共享的可训练参数**。保持 Stage 2.3 和 Stage 3.1 已验收模块的公开行为不变。如果当前现有类的构造参数名称不同，可以在 Block 内部适配；不要求修改已有类的签名。

## 3. 功能需求

### REQ-3.2.1 — Pre-LN 顺序

严格采用以下计算顺序（`+` 为逐元素相加）：

```text
h1 = x + attn(ln1(x))
output = h1 + mlp(ln2(h1))
```

不得将 LayerNorm 放在残差加法之后（Post-LN），也不得把 MLP 输入直接接到未经 Attention 残差更新的 `x`。

### REQ-3.2.2 — 残差与形状

- Attention 分支和 MLP 分支的输出都必须与残差输入形状匹配。
- 残差直接使用当前对应的隐藏状态，不引入额外可训练投影或缩放。
- 输入输出均为 `[B, T, d_model]`。

### REQ-3.2.3 — 独立 LayerNorm

- `ln1`、`ln2` 是两组独立的 `nn.LayerNorm(d_model)`。
- 归一化只针对每个 Token 的最后一个特征维度，不能跨 Batch 或 Token 维度归一化。
- 两个归一化模块均有独立可训练的 affine 参数。

### REQ-3.2.4 — Causal 行为

- 默认 `causal=True`。
- 在 `causal=True` 且 `model.eval()` 的条件下，修改未来 Token 的输入不得改变前缀位置输出。
- `causal=False` 时遵循 Stage 2.3 已验收的双向注意力行为；测试只要求正确传递配置，不强制某一特定输入下的输出必须变化。

### REQ-3.2.5 — 训练/评估与梯度

- `train()` / `eval()` 应按 PyTorch 模块机制传递到子模块。
- MLP Dropout 的行为遵循 Stage 3.1 已验收契约。
- 对代表性输入和损失，输入、Attention、MLP、LayerNorm 参数具有正确的反向传播路径，梯度应为有限值。
- 对于特殊构造导致的零梯度，不应作出“所有元素始终非零”的错误要求。

### REQ-3.2.6 — 兼容性与数值正确性

- 不修改 Stage 1、Stage 2.1、Stage 2.2、Stage 2.3、Stage 3.1 的既有正确性约定。
- 使用固定权重及独立参考计算验证完整 Block 的数学运算，不能只验证 Shape。
- 代表性有限输入与正常初始化下的前向输出及梯度应保持有限。

## 4. 验收标准

- [ ] `TransformerBlock` 接口和构造参数符合约定。
- [ ] `ln1`、`ln2` 为独立的可训练 LayerNorm。
- [ ] `h1 = x + attn(ln1(x))` 的执行顺序正确。
- [ ] `output = h1 + mlp(ln2(h1))` 的执行顺序正确。
- [ ] 不同 Batch、Token 数量下输出形状正确。
- [ ] 固定权重或受控替身模块下，结果与独立手算/参考值一致。
- [ ] 默认因果模式满足前缀不受未来输入影响。
- [ ] 训练/评估模式正确影响 Dropout。
- [ ] 梯度能到达输入和必要参数。
- [ ] 完整历史回归（当前基线 76 项）全部通过。

## 5. 本阶段非目标

- 不实现完整 `MiniGPT` 或 `LM Head`（Stage 3.3）。
- 不新增输入 Tokenizer/Embedding 或位置编码逻辑。
- 不修改 `MultiHeadAttention`、`MLP` 的数学实现来适配测试。
- 不新增 KV Cache、RoPE、FlashAttention、训练循环或专门的混合精度优化。
- 不把实验性的生成质量、训练收敛表现设为本阶段硬断言。

## 6. 阶段交付与职责

**开发者：** `src/transformer.py` 中的 `TransformerBlock` 实现。  
**Codex：** `tests/test_transformer_block.py`，覆盖需求到用例的映射、独立参考数值、回归及问题分类。  

Codex 先按本契约编写测试，不修改生产代码；开发者独立实现；最后由 Codex 只读复验并出具 PASS / FAIL / BLOCKED 报告。

> 本文档是待确认的 Stage 3.2 接口方案。交给 Codex 前先确认与当前 `MLP`、`MultiHeadAttention` 的公开接口兼容；如需调整构造参数名称，应先更新并确认本契约，不能让 Codex自行决定。
