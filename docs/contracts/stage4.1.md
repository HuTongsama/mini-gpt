# Mini-GPT Stage 4.1 — 数据准备与 Next Token Loss

**契约版本：** v0.1  
**状态：** 确认  
**前置条件：** Stage 3.3 PASS（累计 140 项测试通过）  
**目标文件：** `src/dataset.py`、`src/loss.py`

## 1. 目标与边界

为现有 `MiniGPT` 提供可复现的语言模型训练数据和损失计算。Stage 4.1 只准备数据、训练标签和损失函数，不实现优化器、训练循环、采样生成或 Checkpoint。

该阶段将原始文本转换为固定长度的 `(x, y)` 样本，其中 `y` 相对于 `x` 左移一个预测位置：

```text
原始 Token IDs: [10, 11, 12, 13, 14, 15]
context_length = 3
样本 0:
  x = [10, 11, 12]
  y = [11, 12, 13]
样本 1:
  x = [11, 12, 13]
  y = [12, 13, 14]
样本 2:
  x = [12, 13, 14]
  y = [13, 14, 15]
```

训练模型时，`logits[b, t]` 对应 `targets[b, t]` 指定的下一个 Token，而不是当前 Token。

## 2. 公开接口

### 2.1 文本读取与拆分

文件：`src/dataset.py`

```python
from pathlib import Path

def read_text_utf8(path: str | Path) -> str:
    ...

def split_text(text: str, train_fraction: float = 0.9) -> tuple[str, str]:
    ...
```

- `read_text_utf8`：读取 UTF-8 文本；不可静默替换无效编码，文件读取错误向调用方报告。
- `split_text`：按原始字符顺序做连续拆分；`split_index = int(len(text) * train_fraction)`；返回 `text[:split_index]` 和 `text[split_index:]`。
- `train_fraction` 须是有限实数，满足 `0 < train_fraction < 1`；禁止布尔值。结果任一部分为空时明确报错。
- 不随机打乱字符，不让训练集和验证集的样本窗口跨越拆分边界。

### 2.2 NextTokenDataset

文件：`src/dataset.py`

```python
from collections.abc import Sequence
import torch
from torch.utils.data import Dataset

class NextTokenDataset(Dataset):
    def __init__(self, token_ids: Sequence[int], context_length: int):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

契约：

- `token_ids` 是已经编码完成的非负整数 Token ID 序列；不接受 `bool` 作为 Token ID。
- `context_length` 是正整数，不接受 `bool`。
- 需要至少 `context_length + 1` 个 Token，否则明确报错。
- `len(dataset) == len(token_ids) - context_length`。
- 第 `i` 个样本：`x = token_ids[i:i+T]`；`y = token_ids[i+1:i+T+1]`。
- `x` 和 `y` 都是 `torch.long`，形状 `[T]`，不与源数据共享可变 Tensor 存储。
- `__getitem__` 支持 Python 规范范围内的非负索引 `[0, len(dataset))`；越界抛 `IndexError`，不要求支持负索引。
- 与 PyTorch `DataLoader` 配合后，Batch 的形状为 `[B,T]`。

### 2.3 Next Token Loss

文件：`src/loss.py`

```python
import torch

def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    ...
```

- `logits` 形状 `[B,T,V]`，是未经 Softmax 的原始预测分数；`targets` 形状 `[B,T]`，类型 `torch.long`。
- `B >= 1`、`T >= 1`、`V >= 2`；目标值满足 `0 <= targets < V`。
- 使用 **全部 B×T 位置** 的交叉熵，按位置求均值，返回一个标量 Tensor（0 维）。
- 不在 Cross Entropy 前手动应用 Softmax；应使用具有数值稳定性的等价计算。
- 对正常有限输入产生有限 Loss；梯度可以从 Loss 回传到 Logits。
- 输入类型、形状或目标范围不合法时明确报错；不要求固定异常文本，可使用合理的 `TypeError`、`ValueError` 或 PyTorch 异常。

## 3. 需求编号

### REQ-4.1.1 — UTF-8 文本读取

正确读取本地 UTF-8 数据，不静默丢弃或替换字符；非法编码和缺失文件能被明确报告。

### REQ-4.1.2 — 连续训练/验证拆分

连续、不重叠、顺序不变；结果拼接等于原文。切分位置与 `train_fraction` 一致。

### REQ-4.1.3 — Next Token 窗口构造

`x` 与 `y` 严格错位一个位置，所有合法窗口均可访问，最后一个窗口包括语料最后一个 Token。

### REQ-4.1.4 — Dataset 张量、索引和边界

返回 `torch.long`、形状 `[T]`；非法长度、非法 Token ID、非法索引明确失败；Batch 组合正确。

### REQ-4.1.5 — 交叉熵数值正确性

对全部位置计算平均 Next Token Loss；使用独立手算或稳定数学参考值验证；不得只计算最后位置。

### REQ-4.1.6 — 梯度与接口兼容

Loss 对 Logits 可微；能直接接收 Stage 3.3 `MiniGPT` 的 `[B,T,V]` Logits，并与 Dataset 的 `[B,T]` Targets 配合。

## 4. Tokenizer 和数据集集成策略

本阶段复用 Stage 1 的 `CharTokenizer`，不修改其公共接口，不实现 BPE 或 `<UNK>` Token。

建议的数据流：

1. `read_text_utf8` 读取原始文本。
2. `split_text` 将文本分成训练和验证两段。
3. 用**训练文本**调用 `CharTokenizer.fit()`。
4. 用同一词表分别编码训练和验证文本。
5. 分别建立 `NextTokenDataset`；确保两个集合的样本窗口不跨越拆分边界。

**明确的已知限制：** 如果验证文本包含训练词表中从未出现过的字符，沿用 Stage 1 的约定，编码会报错。本阶段不允许偷偷用验证文本扩充训练词表。应明确报告这一情况，或选择覆盖相关字符的训练语料；如需引入 `<UNK>`，应另立需求。

## 5. 验收标准

- [ ] UTF-8 文本读取正确，失败行为明确。
- [ ] 数据划分顺序不变、无重叠、可按拼接还原。
- [ ] Dataset 能覆盖从首个到最后一个合法窗口，无 off-by-one 错误。
- [ ] `x`、`y` 的每个位置满足下一 Token 对齐关系。
- [ ] Tensor 类型、形状、索引、长度边界正确。
- [ ] `next_token_loss` 数值符合独立参考，覆盖多个 Batch 和 Token 位置。
- [ ] 不提前 Softmax；Loss 是标量且梯度可回传。
- [ ] 能与现有 `MiniGPT` 输出契约集成。
- [ ] Stage 1～3.3 的 140 项既有测试全部继续通过。

## 6. 不在本阶段的工作

- 优化器、反向训练循环、梯度裁剪与学习率计划。
- Checkpoint 保存和加载。
- 自动生成文本、采样策略。
- BPE、多模态、KV Cache。
- CUDA、混合精度、分布式训练、性能优化。
- 修改已验收的生产模块以迁就新测试。

## 7. 交付物与验收

开发者负责 `src/dataset.py`、`src/loss.py`。

Codex 负责新增 `tests/test_dataset.py`、`tests/test_loss.py`，以及测试结果报告。测试应先于开发实现，以需求契约为依据。未实现时的新测试失败按 `NOT_IMPLEMENTED` 归类。

验收结果必须区分 Stage 4.1 新测试与历史 140 项回归测试，并注明环境限制。完成后停止，等待 Stage 4.2 契约。
