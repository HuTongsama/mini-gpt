# Mini-GPT Stage 4.2 — 训练循环与验证损失接口契约

**契约版本：** v0.1  
**状态：** 确认  
**前置条件：** Stage 4.1 PASS（累计 209/209 项测试通过）  
**目标生产文件：** `src/train.py`

## 1. 阶段目标与范围

在 Stage 4.1 的 `NextTokenDataset`、`next_token_loss` 与 Stage 3.3 的 `MiniGPT` 之上，实现最小但完整的参数训练闭环：

```text
DataLoader: (x, y) [B, T]
      ↓
model.train() → MiniGPT(x) → logits [B, T, V]
      ↓
next_token_loss(logits, y) → 标量 Loss
      ↓
zero_grad → backward → AdamW.step
      ↓
统计整个 Epoch 的平均 Token Loss
```

另外实现只读的验证损失计算，确保验证阶段不更新模型参数，不使 Dropout 保持在训练模式。

**本阶段不包含**训练入口 CLI、模型保存/加载、学习率调度、采样生成或大规模训练实验；这些不应为完成本契约而提前加入。

## 2. 公开接口

文件：`src/train.py`

```python
import torch
from torch import nn
from torch.utils.data import DataLoader


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> torch.optim.AdamW:
    ...


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: str | torch.device = "cpu",
) -> float:
    ...


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: str | torch.device = "cpu",
) -> float:
    ...
```

说明：这些是 Stage 4.2 新增的公开 API。已有 `MiniGPT`、`NextTokenDataset`、`next_token_loss`、Tokenizer 和 Layer 模块的接口不得为适配训练循环而修改。

### 2.1 `create_optimizer`

- 返回 `torch.optim.AdamW`，管理传入模型的全部可训练参数；不得使用无关模型的参数。
- `learning_rate` 是有限的正实数，`learning_rate > 0`；`weight_decay` 是有限的非负实数，`weight_decay >= 0`。两者均拒绝布尔值及 NaN/Inf。
- 不额外规定 AdamW 的 betas、eps 等参数，使用 PyTorch 默认值。
- 不内置 scheduler、梯度裁剪、参数冻结策略或特殊 parameter group。

### 2.2 `train_one_epoch`

- 输入 `dataloader` 依次产生 `(inputs, targets)`，每个均为 `[B, T]` 的 `torch.long` Tensor，`B >= 1`、`T >= 1`；`targets` 与 `inputs` 已由 Stage 4.1 对齐。
- 模型与优化器由调用方提供，`model` 的参数在调用前应已位于 `device` 对应设备。函数负责将 Batch 的 `inputs`、`targets` 移到 `device`，不负责移动或重建模型/优化器。
- 进入训练阶段时调用 `model.train()`，本函数返回后保留训练模式。
- **对每个 Batch**，必须先清除旧梯度，再调用 `model(inputs)`、`next_token_loss(logits, targets)`、`loss.backward()`、`optimizer.step()`。一次 Batch 对应一次参数更新。
- 梯度清理必须发生在当前 Batch 的反向传播前，以防止不同 Batch 的梯度无意累积。允许 `optimizer.zero_grad()` 或功能等价形式。
- 返回一个 Python `float`，表示**本 Epoch 所有目标 Token 的平均交叉熵**。不能对各 Batch 的平均 Loss 简单等权平均，因为最后一个 Batch 可能较小。
- 统计规则：对第 `k` 个 Batch，`n_k = targets.numel()`，标量 `loss_k` 为该 Batch 所有位置的均值，则返回 `sum(loss_k * n_k) / sum(n_k)`。统计值使用该 Batch **更新参数之前**计算的 Loss。
- 没有任何 Batch 的输入应明确报错（如 `ValueError`），不得返回 NaN 或伪造的 0。
- 正常有限数据、正常初始化下的 Loss 和梯度应保持有限；当检测到非有限的 Batch Loss 时，须在本 Batch `backward()` 和 `step()` 之前明确报错。

### 2.3 `evaluate_loss`

- 使用相同的 Batch 契约与全部 Token 加权平均方式，返回 Python `float`。
- 验证计算必须在 `model.eval()` 与 `torch.no_grad()`（或等价的无梯度上下文）下进行。
- 不得执行 `backward()`、`optimizer.step()`、`zero_grad()` 或更新任何模型参数。
- **保留调用方的模式**：记录进入函数前的 `model.training` 状态，在函数结束时恢复该状态；即使内部因数据或 Loss 出错，也应恢复。若进入时为训练模式，结束时恢复训练模式；若原来是评估模式，保持评估模式。
- 返回计算规则与 `train_one_epoch` 相同；没有任何 Batch 明确报错。
- 非有限 Batch Loss 明确报错；不要求对任意极端有限浮点数都保证数值稳定。

## 3. 功能需求

### REQ-4.2.1 — AdamW 优化器

创建绑定当前模型参数的 AdamW，并正确配置 Learning Rate / Weight Decay；拒绝非法超参数。

### REQ-4.2.2 — 正确训练步骤顺序

每个 Batch 按清梯度、前向、交叉熵、反向、更新参数的顺序运行。训练阶段使用 `model.train()`，正确触发 Dropout。

### REQ-4.2.3 — 参数更新与梯度隔离

在可学习的非退化测试输入上，至少有一个模型参数发生真实变化；不同 Batch 不应无意累积旧梯度。不能只计算 Loss 而漏掉优化步骤。

### REQ-4.2.4 — Epoch Loss 统计

对全部 Token 位置加权平均；特别覆盖尾 Batch 大小不同的情况。返回类型为 Python `float`，而不是仍附带计算图的 Tensor。

### REQ-4.2.5 — 验证阶段只读与模式恢复

验证时执行 `eval()` 与无梯度计算，模型参数保持不变；正确恢复原始训练/评估模式，保证 Dropout 模式切换符合预期。

### REQ-4.2.6 — 数据、错误及模型集成

支持 Stage 4.1 DataLoader 和 Stage 3.3 MiniGPT；空数据明确报错；非有限 Loss 在当前 Batch 更新前报错；已覆盖的正常 CPU `float32` 场景可正确完成训练与验证。

## 4. 参考数据流与学习重点

```text
训练：
  for x, y in train_loader:
      清除旧梯度
      logits = model(x)
      loss = next_token_loss(logits, y)
      反向传播
      AdamW 更新
  返回全部训练 Token 的平均 Loss

验证：
  临时切换到 eval + 无梯度计算
  for x, y in val_loader:
      logits = model(x)
      loss = next_token_loss(logits, y)
  恢复调用前模式
  返回全部验证 Token 的平均 Loss
```

以上是调用顺序说明，**不是生产实现代码**。模型 Loss 不要求每一步都单调下降；更新之后统计的 Loss 也不等同于更新前统计的训练 Loss。

建议的最小集成配置：小词表、小型 `MiniGPT`、短 `context_size`、少量 CPU Batch。大规模拟合及可复现训练实验留给 Stage 4.3。

## 5. 验收标准

- [ ] 公开 API 符合 §2，`create_optimizer` 返回 AdamW。
- [ ] 超参数合法性按契约校验；不接受 NaN、Inf、布尔值等非法值。
- [ ] 每个训练 Batch 的 `zero_grad`、forward、loss、backward、step 顺序正确。
- [ ] 优化步骤实际更新当前模型参数；无跨 Batch 意外梯度累积。
- [ ] 训练与验证统计的是全部 Token 加权平均 Loss，含非满尾 Batch。
- [ ] 训练使用 `train()`；验证使用 `eval()` 与无梯度模式。
- [ ] 验证不修改参数，并在正常或异常结束时恢复原先模式。
- [ ] 空数据、非有限 Batch Loss 有明确失败行为。
- [ ] 与现有 Dataset、Loss 和 MiniGPT 完成一次 CPU `float32` 集成训练/验证。
- [ ] Stage 1～4.1 的历史 **209 项**测试全部继续通过。

## 6. 范围外事项

- Checkpoint 保存、加载和中途恢复（Stage 4.3）。
- 固定训练随机性、正式实验记录、小语料过拟合验收（Stage 4.3）。
- 训练 CLI、配置文件解析、日志系统、训练曲线绘制。
- 自定义 AdamW 或手写反向传播。
- 学习率调度、梯度累积、梯度裁剪、Early Stopping。
- 推理采样与文本生成（Stage 5）。
- CUDA、自动混合精度、分布式训练或性能优化。
- 为了匹配新测试而修改此前已验收的模块行为。

## 7. 交付职责与验收

**开发者：** 独立编写 `src/train.py`。  
**Codex：** 先于开发编写 `tests/test_train.py`，交付需求到测试的映射、独立测试证据和全量回归结果，不修改 `src/`。

若测试先于生产实现，相关新增测试失败标记为 `NOT_IMPLEMENTED`。出现契约歧义应先报告，不自行扩展产品范围。最终复验须报告 Stage 4.2 新测试和历史 209 项回归测试的 PASS/FAIL/SKIP/ERROR，以及运行环境与未覆盖场景。

完成后停止，等待 Stage 4.3 契约。
