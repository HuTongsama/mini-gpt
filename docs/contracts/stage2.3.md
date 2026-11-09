# Mini-GPT Stage 2.3 — Multi-Head Self-Attention 接口契约

**版本：** v0.1  
**状态：** 确认  
**前置条件：** Stage 2.2 PASS（10/10），Stage 2.1 + Stage 1 回归 29/29 PASS  
**目标生产文件：** `src/attention.py`  
**目标测试文件：** `tests/test_multihead_attention.py`

## 1. 阶段目标

在已经掌握 Single-Head Self-Attention 和 Causal Mask 的基础上，实现完整的 Multi-Head Self-Attention：将 Q/K/V 拆分为多个头，各头分别计算 Scaled Dot-Product Attention，拼接头输出，再执行输出投影。

本阶段重点验证维度操作、各头计算、拼接次序、因果性及反向传播。不得以 `torch.nn.MultiheadAttention` 或 `torch.nn.functional.scaled_dot_product_attention` 替代核心实现。

## 2. 公开接口

```python
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        causal: bool = True,
    ):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

### 2.1 参数与形状

| 项目 | 契约 |
|---|---|
| `d_model` | 正整数，输入与输出的特征维度 |
| `num_heads` | 正整数，`d_model % num_heads == 0` |
| `head_dim` | `d_model // num_heads` |
| `causal` | 默认 `True`；`False` 时采用双向 Self-Attention |
| 输入 `x` | float32 Tensor，形状 `[B, T, d_model]`，`B>=1`、`T>=1` |
| 输出 | float32 Tensor，形状 `[B, T, d_model]` |
| 测试环境 | Windows CPU / float32；暂不强制 CUDA 或混合精度 |

为方便独立数值测试，要求模块公开四个可设置权重的 `nn.Linear` 投影属性：`q_proj`、`k_proj`、`v_proj`、`out_proj`。四者均采用 `bias=False`，形状对应 `Linear(d_model, d_model)`。

内部实现可以自主选择张量操作方式，但必须保持公开契约。已有 `SingleHeadAttention` 的接口和行为不允许破坏。

## 3. 功能需求

### REQ-2.3.1：QKV 投影与拆头

对于 `x: [B,T,D]`，计算 Q、K、V，每个先具有 `[B,T,D]` 形状，再按 `H=num_heads`、`d_h=D/H` 拆分为 `[B,H,T,d_h]`。

每个头有自己对应的 Q/K/V 投影参数区域。不得将不同 Head 的序列维度或 Batch 维度错误混合。

### REQ-2.3.2：各头独立计算注意力

每个 Head 按自己的 `d_h` 缩放并计算：

以下使用纯文本公式。对于单个 Batch 中的第 `h` 个 Head：

```text
d_h = head_dim = d_model // num_heads
Q_h、K_h、V_h 的形状均为 [T, d_h]

S_h = (Q_h @ K_h.T) / sqrt(d_h)
A_h = softmax(S_h + M, dim=-1)
O_h = A_h @ V_h

S_h、A_h、M 的形状均为 [T, T]
O_h 的形状为 [T, d_h]
```

其中 `@` 表示矩阵乘法，`K_h.T` 表示该头二维 Key 矩阵的转置；`dim=-1` 表示每个 Query 对所有 Key 的分数独立进行 Softmax 归一化。

Mask 矩阵 `M` 的行索引 `i` 表示 Query 位置，列索引 `j` 表示 Key 位置，索引从 0 开始：

```text
causal=True：
    M[i, j] = 0       当 j <= i（允许访问）
    M[i, j] = -inf    当 j > i （屏蔽未来位置）

causal=False：
    M[i, j] = 0       所有位置均允许访问
```

`-inf` 表示负无穷；Mask 在 Softmax 之前生效。

Attention Scores 和 Weights 应具备 `[B,H,T,T]` 的逻辑形状，按 Key 维度执行 Softmax。无需把这些中间结果作为公开返回值。

### REQ-2.3.3：Causal Mask

- `causal=True`：第 `i` 个 Query 只能关注 `j<=i` 的 Key；屏蔽位置经 Softmax 后权重为零。
- `causal=False`：不对未来位置施加因果屏蔽。
- 同一个掩码语义适用于所有 Batch 和 Head；序列中每一行至少有一个可访问位置。
- 不可因新增多头能力而破坏 Stage 2.2 的因果性要求。

### REQ-2.3.4：拼接与输出投影

将各头 `[B,H,T,d_h]` 的输出按 Head 顺序拼接，恢复 `[B,T,D]`，再经过 `out_proj`，输出 `[B,T,D]`。

最终输出是经过 `out_proj` 的隐藏特征，**不是** `[B,H,T,T]` 的注意力权重。

### REQ-2.3.5：梯度与 Batch 隔离

- 四个投影均参与 PyTorch autograd。
- 代表性非退化输入下，四组投影权重的梯度应为有限值，并可测得非零梯度。
- 不同 Batch 样本的计算互不干扰。
- 因果模式下，仅使用前缀输出计算损失时，未来输入位置不应收到梯度贡献。

### REQ-2.3.6：参数与边界处理

- 对 `d_model<=0`、`num_heads<=0`、非整数参数、`d_model` 不可被 `num_heads` 整除等配置给出明确异常。
- `T=1`、`B=1` 均可正常前向计算。
- 合法且具有可表示中间结果的代表性 float32 输入不得产生 NaN/Inf。
- 不要求对任意有限 float32 输入保证无溢出；保留已发现的 Stage 2.1 数值稳定性测试与实现行为。

## 4. 验收标准

- [ ] 完整单元测试覆盖固定 QKV 权重下的数值结果，参考值独立于生产实现。
- [ ] `num_heads=1` 时，输出与“单头注意力结果再经过指定 `out_proj`”的独立参考计算一致。
- [ ] `num_heads=4`、`d_model=128` 时，输入输出形状都为 `[B,T,128]`，单头维度为 32。
- [ ] 多个 Head 的结果按照约定顺序拼接，测试可发现错位、混头或错误 reshape/transpose。
- [ ] Attention 按 `sqrt(head_dim)` 而不是 `sqrt(d_model)` 缩放。
- [ ] 默认因果模式和显式双向模式符合各自契约。
- [ ] 修改未来 Token 不改变先前输出，并满足相应输入梯度隔离。
- [ ] `T=1`、多个 Batch、长度变化和参数非法场景通过。
- [ ] Q/K/V/输出投影都存在正确的梯度传播路径。
- [ ] Stage 1（16 个）、Stage 2.1（13 个）、Stage 2.2（10 个）原有测试全部通过；不删改以迎合新实现。

## 5. 本阶段不做

- 不修改 Tokenizer、Token Embedding 或 Position Embedding。
- 不实现 Transformer Block、残差连接、LayerNorm 或 MLP。
- 不添加 KV Cache、RoPE、FlashAttention、Dropout 或混合精度优化。
- 不直接使用封装好的 Multi-Head Attention 算子替代核心数学过程。
- 不改变 `SingleHeadAttention` 的公开接口或已验证行为。

## 6. 交付与验收流程

1. **契约确认**：开发者确认本文档的公共接口和测试边界。
2. **Codex 编写测试**：只允许新增/编辑 `tests/test_multihead_attention.py` 及确有必要的测试辅助文件，不得修改任何生产代码。
3. **开发者独立实现**：自行完成 `MultiHeadAttention`。
4. **Codex 复验**：Stage 2.3 新测试 + Stage 2.2/2.1/1 全套回归。
5. **验收报告**：列明按需求 ID 的覆盖、PASS/FAIL/SKIP/ERROR、失败类型、环境与文件变更。
6. **Checkpoint**：通过验收后再提交本阶段代码及测试；Codex 不自动提交或推送。

**阶段通过标准：** 本阶段明确约定的核心测试通过，39 项已有回归测试保持通过；不存在阻断性契约分歧。
