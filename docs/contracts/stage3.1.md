# Mini-GPT Stage 3.1 — MLP 接口契约

> **文档版本：** v0.1  
> **状态：** 确认  
> **前置条件：** Stage 2.3 验收 PASS（累计 55/55 测试）  
> **目标生产文件：** `src/transformer.py`  
> **阶段目标：** 实现供 Transformer Block 使用的、逐 Token 的可训练非线性特征变换模块。

## 1. 范围与非目标

实现两层线性映射、中间 GELU 激活和可配置的输出 Dropout。

MLP 对 Batch 中每个 Token 位置应用**同一组参数**，自身不在不同 Token 位置之间交换信息。

**本阶段不实现：** Transformer Block、LayerNorm、Residual Connection、LM Head、训练循环、Attention 变更或新的位置编码。

## 2. 公开接口

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

### 2.1 参数与输入输出

| 项目 | 契约 |
| :-- | :-- |
| `d_model` | 正整数，输入和输出的特征维度 |
| `hidden_dim` | 正整数，中间隐藏特征维度；v0.1 推荐 `4 * d_model` |
| `dropout` | 概率值，`0 <= dropout < 1`；默认 `0.0` |
| `x` | CPU `float32` Tensor，形状 `[B, T, d_model]`，其中 `B >= 1`、`T >= 1` |
| 返回值 | Tensor，形状 `[B, T, d_model]` |

为了使 Codex 能够进行独立数值测试，模块必须公开两个 `nn.Linear` 成员：`fc1` 和 `fc2`。它们均使用 `bias=True`，测试可向其权重和偏置注入固定值。

## 3. 计算契约

逐 Token 的前向计算顺序为：

```text
输入 X                 [B, T, d_model]
  |
  v
fc1: Linear            [B, T, hidden_dim]
  |
  v
GELU                   [B, T, hidden_dim]
  |
  v
fc2: Linear            [B, T, d_model]
  |
  v
Dropout(p=dropout)     [B, T, d_model]
  |
  v
输出                   [B, T, d_model]
```

其中 GELU 使用 PyTorch 默认的精确模式：`approximate="none"`。

按 PyTorch `nn.Linear` 的权重存储约定，其数学表达式为：

```text
H = GELU(X @ fc1.weight.T + fc1.bias)
Y = Dropout(H @ fc2.weight.T + fc2.bias)
```

说明：`@` 表示对 Tensor 最后一维执行线性映射，偏置通过 Broadcasting 添加。上式仅表达运算含义，并非要求使用这些语句作为生产实现。

**特别说明：** 本模块不包含 LayerNorm 或残差相加；这些由 Stage 3.2 的 Transformer Block 负责。

## 4. 功能需求

### REQ-3.1.1 — 维度与逐位置处理

- 输入 `[B, T, D]`，输出 `[B, T, D]`；`B`、`T` 可以变化。
- 所有 Token 位置共享 MLP 参数。
- MLP 本身不得在序列维度混合数据。

### REQ-3.1.2 — 可训练投影

- 公开 `fc1`、`fc2` 两个独立、可训练的 `nn.Linear`。
- 两层均具有 Bias。
- 允许测试向权重和偏置注入固定值。

### REQ-3.1.3 — 非线性运算

- 在 `fc1` 和 `fc2` 之间使用 `GELU(approximate="none")`。
- 不得省略激活函数，也不得用 ReLU 替代。

### REQ-3.1.4 — Dropout

- 默认 `dropout=0.0`。
- 训练模式使用 `nn.Dropout` 语义。
- `eval()` 模式不进行随机丢弃。
- Dropout 位于 `fc2` 之后。

### REQ-3.1.5 — 自动微分

代表性正常输入下，输入、`fc1`、`fc2` 的权重和偏置均应参与计算图，并能获得数值有限的梯度。

### REQ-3.1.6 — 参数有效性

对于非正整数的 `d_model`、`hidden_dim`，以及不在 `[0, 1)` 范围内的 `dropout`，必须明确报错。

## 5. 验收标准

- [ ] `MLP(128, 512, dropout=0.0)` 接收 `[2, 16, 128]`，输出 `[2, 16, 128]`。
- [ ] 固定权重和 Bias 下，输出与独立数值参考一致（允许合理的浮点容差）。
- [ ] 能验证 GELU 确实执行，且操作顺序正确。
- [ ] 在关闭 Dropout 时，相同输入位置的输出一致；只修改一个 Token 位置，不会影响其他位置的输出。
- [ ] `eval()` 下，`dropout > 0` 的输出符合无随机丢弃的参考结果；`train()` 下符合 Dropout 语义。
- [ ] 梯度能从输出传播到输入以及两层 Linear 的权重和偏置。
- [ ] 非法构造参数被拒绝；正常范围内结果无 NaN / Inf。
- [ ] Stage 1、Stage 2.1、Stage 2.2 和 Stage 2.3 的原有 **55 个测试**继续通过。

**数值边界说明：** 本阶段不要求对任意极大但有限的 `float32` 输入都返回有限输出，除非进一步明确中间计算的可表示范围；此限制不能用于豁免常规输入下的非有限结果。

## 6. 交付物与开发流程

| 角色 | 交付内容 |
| :-- | :-- |
| 开发者 | `src/transformer.py` 中的 `MLP` 类（不得提前实现完整 Block） |
| Codex | `tests/test_mlp.py`、测试覆盖表、测试结果和失败分类报告 |

如果决定将 MLP 独立放在 `src/mlp.py`，应先更新本契约和 Codex 测试任务中的目标路径，再开始编写测试。

开发顺序：

1. 开发者确认接口契约。
2. Codex 根据契约先编写测试，不修改生产代码。
3. 开发者独立实现 MLP。
4. Codex 执行 Stage 3.1 测试与历史回归。
5. 审阅验收结果，通过后建立 Git Checkpoint。

**权限边界：** 未经开发者确认，Codex 不得修改本契约或生产代码。
