# Mini-GPT Stage 2.2 — Causal Mask 接口契约

- **契约版本**：v0.1
- **状态**：确认
- **前置条件**：Stage 2.1 PASS（13/13）；Stage 1 回归 PASS（16/16）
- **目标生产文件**：`src/attention.py`
- **建议测试文件**：`tests/test_causal_attention.py`

## 1. 阶段目标与范围

在 Stage 2.1 已完成的 `SingleHeadAttention` 上增加可选的因果注意力约束。启用时，序列中第 `i` 个 Query 位置只能访问第 `0..i` 个 Key 位置；不启用时，保持 Stage 2.1 原有的双向 Self-Attention 行为。

本阶段**不实现** Multi-Head Attention、Transformer Block、KV Cache、位置编码或新的输出投影层。不改变 Stage 2.1 既有 QKV 投影、缩放和聚合数学定义。

## 2. 公开接口

```python
class SingleHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        causal: bool = False,
    ):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

| 条目 | 约束 |
| --- | --- |
| `d_model` | 正整数；输入特征维度 |
| `head_dim` | 正整数；单头 Q、K、V 特征维度 |
| `causal` | 布尔值，默认 `False`；控制是否启用因果注意力 |
| `x` | 形状 `[B, T, d_model]` 的 `float32` Tensor；本阶段主要验收在 Windows CPU 上进行 |
| `B`, `T` | 正整数（`B >= 1`, `T >= 1`） |
| 返回值 | 形状 `[B, T, head_dim]` 的 Tensor |

保留已公开的 `q_proj`、`k_proj`、`v_proj`，其接口为独立可训练的无偏置线性投影；测试可向这些投影注入固定权重。`causal=False` 时必须保持 Stage 2.1 的可观察行为。错误输入形状及异常消息不在本阶段规定的新增验收范围内。

## 3. 数学定义

对于单个 Batch 中的长度为 `T` 的序列，以下使用纯文本公式，`@` 表示矩阵乘法，`K.T` 表示二维矩阵转置：

```text
d_k = head_dim
Q、K、V：形状均为 [T, d_k] 的实数矩阵

S = (Q @ K.T) / sqrt(d_k)
S 的形状为 [T, T]
```

当 `causal=True`，定义形状为 `[T, T]` 的 Mask 矩阵 `M`。行索引 `i` 表示 Query 位置，列索引 `j` 表示 Key 位置，索引从 0 开始：

```text
M[i, j] = 0       当 j <= i（当前位置及过去位置，允许访问）
M[i, j] = -inf    当 j > i （未来位置，禁止访问）

A = softmax(S + M, dim=-1)
O = A @ V

A 的形状为 [T, T]
O 的形状为 [T, d_k]
```

其中 `-inf` 表示负无穷；`dim=-1` 表示沿最后一维，即每个 Query 对所有 Key 的分数独立进行 Softmax 归一化。

当 `causal=False`：不施加因果遮挡，相当于 `M = 0`。

**重要约束**：Mask 必须在 Softmax 归一化之前生效，确保未来位置的注意力权重恰为 0，允许访问的位置权重之和为 1。具体实现方式不限定为与 `-inf` 相加，只要结果等价即可。保持 Stage 2.1 已验证的浮点缩放边界行为。

## 4. 功能需求

### REQ-2.2.1 — 向后兼容

- 默认 `causal=False`。
- 非因果模式保留 Stage 2.1 的 QKV、Scores、Softmax、Value 聚合语义及数值行为。
- 原 Stage 2.1 的 13 个测试不修改、不削弱，必须继续通过。

### REQ-2.2.2 — 因果访问限制

- 启用 `causal=True` 时，位置 `i` 仅能访问 `j <= i` 的 Key/Value。
- 位置 0 只访问位置 0；最后一个位置允许访问整个前缀（包括自身）。
- 不允许未来 Token 影响先前 Token 的 Attention 输出。

### REQ-2.2.3 — 权重归一化与数值正确性

- 被屏蔽位置的注意力权重为 0。
- 每个 Query 对其允许访问的位置做 Softmax 归一化，权重和约为 1。
- 对固定 QKV 投影权重和代表性输入，最终输出与独立参考计算一致。
- 不要求生产接口返回 Attention Weights；可通过精心设计的固定 Value 和输出检验权重行为。

### REQ-2.2.4 — 前缀不变性

- 对相同的输入前缀，修改任意未来位置的 Token 特征后，前缀位置的输出应保持一致（在约定浮点容差内）。
- 本阶段无 Dropout，保证该性质的测试不受随机性干扰。

### REQ-2.2.5 — 形状及边界

- 验证 `B=1`、`B>1`；`T=1`、`T=2`、`T>2`。
- 对所有合法形状，输出保持 `[B,T,head_dim]`。
- 不同 Batch 样本互不影响。

### REQ-2.2.6 — 梯度传播与因果性

- 因果模式下三个 QKV 投影仍可通过自动微分得到梯度。
- 在代表性非退化输入下，梯度应有限。
- 位置 `i` 的输出或仅依赖该位置输出的标量损失，对未来输入位置 `j>i` 的梯度应为 0（在数值容差内）。
- 不要求每个测试情形中全部投影参数梯度都非零：例如 `T=1` 时部分梯度可能数学上为 0。

## 5. 验收标准

- [ ] `causal=False` 与 Stage 2.1 行为兼容。
- [ ] `causal=True` 正确实施下三角访问约束。
- [ ] 因果模式中的未来 Attention 权重为 0，允许位置重新归一化。
- [ ] 固定权重输入下的输出符合独立计算的数学预期。
- [ ] 修改未来输入不改变此前位置的输出。
- [ ] 输入梯度不存在由未来位置向过去输出产生的非法依赖。
- [ ] `B=1`、`B>1`、`T=1`、`T=2`、`T>2` 的输出形状与 Batch 隔离正确。
- [ ] QKV 投影可训练，反向传播正确。
- [ ] Stage 2.2 新测试通过，Stage 2.1 原 13 个测试与 Stage 1 原 16 个测试全部回归通过。

## 6. 测试与交付边界

- **开发者**：仅负责独立实现 `src/attention.py` 中的 Stage 2.2 生产逻辑。
- **Codex**：先依据本契约设计并实现 `tests/test_causal_attention.py`，再于开发完成后运行测试并报告结果；不得修改 `src/` 生产代码。
- 测试应使用固定权重、可独立核验的数值及数学不变量；不得以生产实现的输出作为期望结果。
- 初次生成测试时，`causal` 尚未实现造成的失败应归类为 `NOT_IMPLEMENTED`，不能视为测试交付失败。
- 保留既有测试，不得为了通过而修改已确认的断言；不得自行 Git Commit 或 Push。
- 交付报告应列出：需求覆盖、PASS/FAIL/SKIP/ERROR、失败分类、修改文件、是否满足阶段验收。

## 7. 阶段结束条件

新增 Stage 2.2 测试和此前 29 个回归测试全部通过，且无未解释的契约偏离时，由开发者确认 Stage 2.2 PASS，建立 Git Checkpoint，再进入 Stage 2.3 Multi-Head Attention。
