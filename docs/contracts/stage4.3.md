# Mini-GPT Stage 4.3 — Checkpoint 与小样本学习验证接口契约

**契约版本：** v0.1  
**状态：** 确认  
**前置条件：** Stage 4.2 PASS（累计 295/295 项测试通过）  
**目标生产文件：** `src/checkpoint.py`

## 1. 阶段目标与范围

Stage 4.2 已实现 `create_optimizer()`、`train_one_epoch()` 与 `evaluate_loss()`。本阶段需要验证两件事：

1. **保存与恢复：** 在 Epoch 边界保存模型参数、AdamW 优化器状态和 CPU 随机数状态；加载后能够从已完成的 Epoch 数继续训练，而非重新开始。
2. **实际学习：** 在固定、极小的字符级数据上，通过真实 `MiniGPT → Loss → backward → AdamW` 训练闭环，使重新评估得到的 Loss 明显下降，证明不仅可以更新参数，而且模型确实能够拟合训练样本。

本阶段不新建训练算法；复用已验收的训练、数据与模型模块。小样本拟合只要求**学习实验与独立测试证据**，不要求新建复杂生产训练入口。

## 2. 公开 API

文件：`src/checkpoint.py`

```python
from pathlib import Path
import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    epoch: int,
) -> None:
    ...


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> int:
    ...
```

- `epoch` 表示**已经完整执行的 Epoch 数量**。初始状态是 `epoch=0`；完成首轮训练并保存时使用 `epoch=1`。必须是非负 Python 整数，不接受 `bool`。
- `path` 接受字符串或 `pathlib.Path`，指向一个文件。父目录必须已经存在；本阶段不要求自动创建目录。
- `model` 和 `optimizer` 由调用方创建、绑定和放置到合适的设备；Checkpoint 模块**不得自行实例化模型、重新创建优化器或调用 `model.to()`**。
- `optimizer=None` 仅用于**只恢复模型权重**的场景（方便后续 Stage 5 推理）；保存时始终需要优化器。
- `load_checkpoint` 返回文件记录的 `epoch`，并恢复对应权重；它不执行训练、不自动切换 `model.train()` / `model.eval()`，也不返回新模型对象。
- 本阶段要求 Windows CPU、`float32`；`map_location` 用于加载时的张量映射，不代表自动迁移调用方模型。

## 3. Checkpoint 数据契约

首版保存以下明确字段，使用 PyTorch 可安全加载的基础值与 Tensor / `state_dict`：

```text
format_version: 1
completed_epochs: 非负整数
model_state: model.state_dict()
optimizer_state: optimizer.state_dict()
cpu_rng_state: torch.get_rng_state()
```

- `model_state` 必须保存参数和持久 Buffer，而不是只保存某些层的权重。
- `optimizer_state` 必须保留 AdamW 的状态（如动量估计、步数）与 parameter groups；只保存学习率或重新创建优化器都不符合契约。
- `cpu_rng_state` 用于恢复 PyTorch **CPU** 随机数生成器状态。Checkpoint 在 Epoch 边界使用：此时没有进行中的 Batch。
- 文件不包含模型对象的 Python pickle 实例、不包含自定义 Python 类实例。读取时使用适用的安全载入方式（例如支持的 PyTorch 版本下 `torch.load(..., weights_only=True)`），不得以不可信文件进行任意对象反序列化。
- `format_version` 是 Checkpoint **文件格式标识**，与本文档 `v0.1` 的文档版本不同。
- 本阶段不保存 Tokenizer 本身；Stage 1 已有的 Tokenizer 词表持久化机制仍由调用方独立使用。模型加载时必须由调用方创建结构兼容的实例，并使用匹配的词表。

## 4. 功能需求

### REQ-4.3.1 — 保存完整训练状态

- `save_checkpoint` 将 §3 指定的全部字段写入 `path`，成功后文件存在且可读取。
- 保存不修改模型参数、优化器状态、训练/评估模式或 CPU RNG 状态。
- 文件路径不可写或父目录不存在时明确失败；非法 `epoch` 在写入前拒绝。
- 不要求跨操作系统、跨 PyTorch 版本或任意精度的二进制格式兼容。

### REQ-4.3.2 — 模型权重恢复

- 在相同架构的全新 `MiniGPT` 上，加载成功后所有 `state_dict` 条目与保存时对应。
- 对同一输入、`eval()` 模式且关闭随机性时，保存前与加载后的 Logits 数值一致（CPU `float32` 在测试容差内）。
- `optimizer=None` 时只恢复模型状态，不要求重建或恢复优化器；仍返回已完成的 Epoch 数。
- 不要求为模型不兼容的情况提供参数自动迁移；必须明确失败，不得静默忽略缺失/多余权重。

### REQ-4.3.3 — AdamW 状态恢复与后续更新

- 先完成至少一次真实 AdamW 更新，再保存。
- 新建结构相同的模型和绑定其参数的优化器，载入后恢复原有优化器状态、步数和参数组配置。
- 继续训练时，恢复路径与不中断参考路径在相同 Batch 顺序和数值条件下产生相符的模型参数；不能仅恢复模型权重却丢弃 AdamW 历史。
- 无需恢复 `param.grad`：下一次训练调用将按 Stage 4.2 契约在反向传播前清除旧梯度。

### REQ-4.3.4 — CPU RNG 与 Epoch 进度恢复

- `load_checkpoint(..., restore_rng=True)` 恢复保存的 `torch.get_rng_state()`；`False` 时保持调用方当前 RNG 状态不变。`restore_rng` 只能是布尔值。
- `load_checkpoint` 返回正确的 `completed_epochs`，便于调用方确定下一轮编号。
- 用固定顺序的 Batch（例如 `shuffle=False`）及相同模型/优化器训练，从 Epoch 边界中断并恢复时，后续更新应与不中断运行数值相符；可以使用非零 Dropout 检查 RNG 恢复。
- 这里的复现保证只适用于相同 CPU 环境、相同输入顺序及相同操作序列；**不覆盖**额外的 DataLoader 独立 Generator、Python/NumPy RNG、随机数据增强或多工作进程采样器状态。

### REQ-4.3.5 — 错误文件和非法输入

- 对不存在的文件、损坏或缺少必需字段的文件、`format_version` 不支持、负数/布尔型 Epoch、与模型结构不匹配的权重，明确抛出合理异常。
- 若要求恢复优化器但文件无有效优化器状态，不得静默接受。
- 不要求异常加载过程具备事务性回滚（调用方应避免在不可信或无效文件上复用重要的在训实例）；但不得把不完整加载报告为成功。
- 不固定异常消息或具体异常子类，只要求具有可判断的失败行为。

### REQ-4.3.6 — 小样本学习实验

在 Windows CPU `float32` 环境，使用真实的 `MiniGPT`、`NextTokenDataset`、`DataLoader`、`next_token_loss` 和 Stage 4.2 训练 API：

1. 固定 `torch.manual_seed`，使用字符种类很少、规律明确的训练语料。建议重复交替序列，例如 `"abababab..."`，确保训练词表覆盖全部字符。
2. 使用明显小于默认配置的教学模型（如 `embedding_dim=16`、`num_layers=1`、`num_heads=2`、`mlp_hidden_dim=32`、短上下文），使测试运行时间可控。
3. 设置 `dropout=0.0`、固定 `shuffle=False`，在**同一个训练数据集**上分别用 `evaluate_loss()` 测量训练前与若干轮优化后的 Loss。
4. 通过固定种子与足够训练步数，提供可复现证据，要求训练后 Loss **小于训练前的 80%**。这是小样本**训练集拟合能力**测试，而非泛化能力验证；不要求每一步、每轮训练 Loss 单调下降。
5. 至少演示一次保存并恢复模型状态，再验证恢复后能继续训练。Checkpoint 往返测试与小样本 Loss 降低可使用独立的测试数据与配置。

测试不应使用验证集帮助调整词表，不要求生成可读文本；对真实语料的泛化性能和验证集指标研究留待后续实验。

## 5. 学习重点与推荐实验顺序

```text
小样本：固定 Token IDs / Dataset
            ↓
      评估初始 Loss
            ↓
    train_one_epoch × N
            ↓
      评估训练后 Loss
            ↓
     保存 Checkpoint
            ↓
    新建模型 + 优化器
            ↓
      加载并复查参数
            ↓
        继续训练
```

区分三类量：

- `train_one_epoch()` 返回的是**该轮更新前**各 Batch Loss 的 Token 加权平均。
- `evaluate_loss()` 得到的是**当前模型参数下**重新计算的 Loss，适合用于比较训练前后拟合程度。
- 模型权重与 AdamW 状态分别决定下一步更新；仅仅能恢复相同 Logits，不代表训练可以无缝续接。

## 6. 验收清单

- [ ] 公开函数签名及其错误处理符合 §2。
- [ ] Checkpoint 字段、格式版本、模型与优化器状态完整且可以加载。
- [ ] 加载到全新、结构一致的 MiniGPT 后模型 `state_dict` 和 Logits 与保存前一致。
- [ ] AdamW 步数、状态与 parameter groups 恢复；后续训练和不中断参考在固定条件下相符。
- [ ] CPU RNG 与已完成 Epoch 数正确恢复；`restore_rng=False` 不改变当前 RNG。
- [ ] 不存在、损坏、字段缺失、版本错误和结构不匹配均明确失败。
- [ ] 固定小语料上，训练后 `evaluate_loss` 小于初始值的 80%。
- [ ] Checkpoint 保存/加载不会提前实现自动采样、训练 CLI 或依赖升级。
- [ ] Stage 1～4.2 历史 **295 项**测试全量回归通过。

## 7. 非目标与职责边界

本阶段**不实现**：

- 新的训练算法、学习率调度、梯度裁剪、混合精度、分布式训练。
- 训练 CLI、配置系统、自动绘图、实验追踪平台、大规模语料训练。
- Batch 中途恢复、DataLoader shuffle 生成器状态、跨设备严格位级复现。
- Tokenizer 自动打包进 Checkpoint、压缩/加密格式、旧格式迁移。
- 自回归 `generate()`、Temperature、Top-k、KV Cache（Stage 5）。
- 为通过本阶段测试而改写 Stage 1～4.2 的已验收公开契约。

**开发者：** 独立实现 `src/checkpoint.py`，完成符合 §6 的本地验证。  
**Codex：** 测试先行，仅新增 `tests/test_checkpoint.py`、`tests/test_overfit.py`（及必要测试专用辅助文件），进行独立验证，不修改生产代码或历史测试、不提交或推送。

若新生产文件尚不存在，相关新测试失败应标记为 `NOT_IMPLEMENTED`；若契约与现有代码存在重大冲突，应报告 `CONTRACT_AMBIGUITY`，不擅自调整接口。Codex 测试交付并不等于实现验收。
