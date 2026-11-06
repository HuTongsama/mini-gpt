# Mini-GPT Stage 2.1 — Single-Head Self-Attention

**状态：** 确认  
**前置条件：** Stage 1 PASS  
**目标文件：** `src/attention.py`

## 1. 阶段目标

实现一个单头 Self-Attention 模块，完成 QKV 投影、Attention Scores 计算、Softmax 归一化和 Value 加权聚合。

本阶段不涉及 Multi-Head Attention、Causal Mask、Dropout 或 Transformer Block。

## 2. 公开接口

```python
class SingleHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int,
    ):
        ...

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

### 参数契约

| 参数 | 含义 | 约束 |
|---|---|---|
| `d_model` | 输入特征维度 | 正整数 |
| `head_dim` | QKV 特征维度 | 正整数 |
| `x` | 输入隐藏状态 | `[B, T, d_model]` |
| 返回值 | Attention 输出 | `[B, T, head_dim]` |

要求 `B >= 1`、`T >= 1`，本阶段测试使用 CPU、float32。

## 3. 功能要求

### REQ-2.1.1：QKV 投影

模块必须具备三个独立且可训练的投影：

- `q_proj`
- `k_proj`
- `v_proj`

为方便独立数值测试，约定三者为公开的 `nn.Linear` 模块，使用 `bias=False`。

输出 Q、K、V 形状均为：

`[B, T, head_dim]`

测试允许向三个投影层注入固定权重。

### REQ-2.1.2：Scaled Dot-Product

使用 Q 和 K 计算 Attention Scores，并按照 `sqrt(head_dim)` 缩放。

Scores 形状为：

`[B, T, T]`

### REQ-2.1.3：Softmax

沿 Key 对应的维度进行 Softmax 归一化。

每个 Query 对所有 Key 的权重之和应为 1，允许合理浮点误差。

### REQ-2.1.4：Value 聚合

使用 Attention 权重对 V 进行加权求和。

返回：

`[B, T, head_dim]`

不得将 Attention 权重矩阵直接作为最终返回值。

### REQ-2.1.5：梯度传播

三个投影层的参数必须参与 PyTorch 自动微分。

允许模型通过 `loss.backward()` 更新参数梯度。

## 4. 验收标准

- [ ] 输入输出形状正确。
- [ ] QKV 为三个独立的可训练投影。
- [ ] 固定权重下，Attention 数值与独立参考计算一致。
- [ ] 正确使用 `sqrt(head_dim)` 缩放。
- [ ] Softmax 沿正确维度执行。
- [ ] 单 Token 输入正常处理。
- [ ] 多 Batch 输入互不干扰。
- [ ] QKV 参数能够获得梯度。
- [ ] 无效的 `d_model` 和 `head_dim` 被明确拒绝。
- [ ] 合法输入下不出现 NaN 或 Inf。

## 5. 本阶段禁止事项

- 不实现 Causal Mask（留给 Stage 2.2）。
- 不实现 Multi-Head Attention（留给 Stage 2.3）。
- 不添加 Position Embedding（输入已由上一阶段生成）。
- 不实现 KV Cache。
- 不直接使用 `nn.MultiheadAttention` 或 `scaled_dot_product_attention` 代替核心计算。
- 不引入额外的输出投影层。

## 6. 阶段交付

开发者交付：

`src/attention.py`

Codex 交付：

`tests/test_attention.py`

测试运行结果与验收报告。

**验收通过后才能进入 Stage 2.2。**