# Mini-GPT 从零实现项目需求文档

**文档版本：** v0.1  
**项目名称：** Mini-GPT  
**项目类型：** 教学型深度学习工程  
**技术栈：** Python 3.12+、PyTorch、pytest  
**开发策略：** 分阶段实现、单元测试、独立验收

---

## 1. 项目目标

### 1.1 核心目标

从随机初始化参数开始，独立实现一个基于 Decoder-only Transformer 的小型自回归语言模型。

完成后，模型应当具备以下能力：

1. 将原始文本转换为 Token ID 序列。
2. 将 Token ID 转换为可训练的 Embedding。
3. 利用 Self-Attention 建立 Token 之间的信息关联。
4. 利用多层 Transformer 处理上下文。
5. 通过 Next Token Prediction 进行训练。
6. 根据给定 Prompt 自回归生成文本。

### 1.2 非目标

第一版本不追求：

- 达到 ChatGPT 的语言能力。
- 使用大规模预训练数据集。
- 支持多模态。
- 实现分布式训练或 CUDA Kernel。
- 支持 KV Cache 和推理服务部署。
- 使用 RLHF、DPO 等后训练技术。

项目优先保证**实现正确、结构清晰、结果可验证**，其次才考虑性能优化。

## 2. 基础架构约束

### 2.1 模型参数

| 参数 | 初始配置 |
|---|---|
| Tokenizer | Character-level |
| Context Length | 64 |
| Embedding Dimension | 128 |
| Transformer Layers | 2 |
| Attention Heads | 4 |
| Head Dimension | 32 |
| MLP Hidden Dimension | 512 |
| Batch Size | 32 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Architecture | Decoder-only, Pre-LN |
| Training Objective | Next Token Prediction |

这些是初始实验参数，不是要求模型必须使用的固定值。除维度约束外，应当尽量通过配置文件调整。

### 2.2 实现原则

允许直接使用 PyTorch 的：

- `torch.Tensor`、`torch.matmul`
- `nn.Linear`、`nn.Embedding`
- `nn.LayerNorm`、`nn.GELU`
- `torch.autograd`、`torch.optim`
- `torch.nn.functional.cross_entropy`

核心 Attention 计算必须自行实现，第一版不使用 `nn.MultiheadAttention` 或完整的 `nn.Transformer` 作为替代。

不加载任何预训练权重。

---

# 阶段一：Tokenizer 与 Embedding

## 3. 阶段目标

建立从文本到 Transformer 输入矩阵的完整转换流程。

### 3.1 功能需求

**REQ-1.1：字符级 Tokenizer**

实现 `CharTokenizer`，支持：

- `fit(text)`：从训练语料建立字符词表。
- `encode(text)`：将字符串转换为 Token ID。
- `decode(ids)`：将 Token ID 转换回字符串。
- `vocab_size`：返回词表大小。
- 词表的保存和加载。

初始词表应保证字符与 Token ID 一一对应。对于词表以外的字符，第一版应明确报错，不允许静默丢弃。

**REQ-1.2：Token Embedding**

实现基于 `nn.Embedding` 的向量映射。

输入：

`Token IDs: [B, T]`

输出：

`Token Embedding: [B, T, D]`

其中：

- B：Batch Size
- T：Sequence Length
- D：Embedding Dimension

**REQ-1.3：Position Embedding**

使用可训练的位置 Embedding，允许每个位置具有不同的向量。

最终输入表示：

`H0 = TokenEmbedding + PositionEmbedding`

要求输出形状仍为 `[B, T, D]`。

### 3.2 验收标准

- [ ] 任意词表内文本均满足 `decode(encode(text)) == text`。
- [ ] 字符到 Token ID 的映射在保存及加载后保持一致。
- [ ] 输入 `[2, 16]`，Embedding 输出为 `[2, 16, 128]`。
- [ ] 相同 Token ID 对应相同的基础 Token Embedding。
- [ ] 不同位置能够得到不同的位置表示。
- [ ] 超出上下文长度时能够正确报错。
- [ ] Embedding 参数能够通过反向传播获得梯度。

### 3.3 交付物

`tokenizer.py`、`embedding.py`、`tests/test_tokenizer.py`、`tests/test_embedding.py`

**本阶段不涉及 Attention、Transformer 或训练循环。**

---

# 阶段二：Self-Attention 与 Multi-Head Attention

## 4. 阶段目标

独立实现 Transformer 中最核心的 Attention 运算，理解 QKV 和 Token 间的信息交互。

### 4.1 功能需求

**REQ-2.1：QKV 线性映射**

实现三个可训练的线性投影：

`Q = X @ Wq`

`K = X @ Wk`

`V = X @ Wv`

必须正确处理 Batch 和 Sequence 维度。

**REQ-2.2：Scaled Dot-Product Attention**

实现：

`Scores = Q @ K.transpose(-2, -1) / sqrt(head_dim)`

对 Scores 的最后一维执行 Softmax，再与 V 相乘，得到输出。

**REQ-2.3：Causal Mask**

实现因果掩码。

第 i 个 Token 只能关注位置不大于 i 的 Token，不允许使用未来位置的信息。

**REQ-2.4：Multi-Head Attention**

将 Embedding 分为 4 个 Attention Head，每个 Head 独立计算 Attention。

要求实现：

1. QKV 投影。
2. Head 维度拆分。
3. 各 Head 的 Attention 计算。
4. Head 结果拼接。
5. 输出线性投影。

### 4.2 关键验收标准

- [ ] Q、K、V 维度满足设计要求。
- [ ] Attention Scores 形状为 `[B, H, T, T]`。
- [ ] 每行 Softmax 权重之和约等于 1。
- [ ] 未来位置的 Attention 权重为 0。
- [ ] 修改未来 Token，不会改变此前位置的 Attention 输出（测试时关闭 Dropout）。
- [ ] Multi-Head Attention 输出形状与模型隐藏状态一致。
- [ ] QKV 及输出投影参数均可接收梯度。
- [ ] 对固定的小规模输入，计算结果与独立参考实现一致。

### 4.3 交付物

`attention.py`、`tests/test_attention.py`

额外输出一组固定输入下的 Q、K、V、Attention Scores、Softmax 权重和最终输出，作为原理学习记录。

**本阶段暂不实现 KV Cache 或任何 Attention 加速算法。**

---

# 阶段三：完整 Transformer Block 与 GPT 模型

## 5. 阶段目标

将 Attention、MLP、LayerNorm 和残差连接组装成完整的 Decoder-only Transformer。

### 5.1 功能需求

**REQ-3.1：MLP**

实现两层前馈网络：

`Linear(128, 512) → GELU → Linear(512, 128)`

可以增加 Dropout。

**REQ-3.2：Transformer Block**

使用 Pre-LN 结构：

`H1 = X + Attention(LayerNorm(X))`

`H2 = H1 + MLP(LayerNorm(H1))`

要求 Attention 与 MLP 均通过残差连接更新隐藏状态。

**REQ-3.3：GPT 模型**

实现 `MiniGPT`：

`Token IDs → Embedding → Transformer Blocks → Final LayerNorm → LM Head`

其中：

- Transformer Blocks 数量为 2。
- LM Head 将 128 维隐藏状态映射到词表大小。
- 模型在训练前采用随机参数初始化。
- 第一版可以不共享 Token Embedding 与 LM Head 权重。

### 5.2 输入输出契约

输入：

`idx: [B, T]`

输出：

`logits: [B, T, V]`

其中 V 是词表大小。

这里需要注意，输出是每个位置的 Logits，而不是直接返回概率或生成后的字符串。

### 5.3 验收标准

- [ ] 模型能够完成一次前向传播。
- [ ] 输出维度为 `[B, T, vocab_size]`。
- [ ] 模型支持不同的 T，且 T 不超过 Context Length。
- [ ] 所有可训练模块均有正确的梯度传播路径。
- [ ] 输出不存在 NaN 或 Inf。
- [ ] 固定随机种子和输入，在关闭随机性后能够复现相同输出。
- [ ] 修改后续 Token 不影响此前位置的 Logits。

### 5.4 交付物

`transformer.py`、`model.py`、`tests/test_transformer.py`、`tests/test_model.py`

**本阶段完成后，应当已经有一个结构完整、但尚未训练的 GPT。**

---

# 阶段四：训练系统

## 6. 阶段目标

让随机初始化的模型通过预测下一个 Token 的任务学习文本规律。

### 6.1 功能需求

**REQ-4.1：训练数据处理**

实现：

- 读取本地 UTF-8 文本。
- 使用训练集构建 Tokenizer。
- 将文本转换成 Token ID。
- 划分训练集与验证集。
- 构建固定长度的训练样本。

训练样本必须采用错位一位的方式构建：

`X = tokens[i : i + T]`

`Y = tokens[i + 1 : i + T + 1]`

训练和验证窗口不得跨越两个数据集的边界。

**REQ-4.2：Loss 计算**

使用 Cross Entropy Loss。

要求模型在所有有效位置上预测下一个 Token，而不是只在序列最后一个位置计算损失。

不应在传给 PyTorch Cross Entropy 之前手动执行 Softmax。

**REQ-4.3：训练循环**

支持：

- Forward
- Loss
- Backward
- Optimizer Step
- 梯度清零
- 梯度裁剪
- 训练日志记录
- Validation Loss 计算

**REQ-4.4：Checkpoint**

至少保存：

- 模型权重。
- 模型超参数。
- Tokenizer 词表及必要配置。

允许从 Checkpoint 加载模型进行推理。

恢复训练所需的优化器状态可以作为扩展要求。

### 6.2 验收分两步

**验收 A：小样本过拟合**

先构造一个非常小、可重复的文本数据集。

要求：

- [ ] 模型能够正常反向传播。
- [ ] 训练 Loss 明显下降。
- [ ] 可以接近记住小样本的下一个 Token 映射。
- [ ] 在关闭 Dropout 后，相同数据可以稳定复现实验结果。

建议以 Token 级训练准确率超过 95% 作为调试目标，具体阈值可按样本调整。

**验收 B：正常文本训练**

使用较大的中文文本语料训练：

- [ ] Training Loss 有持续下降的趋势。
- [ ] Validation Loss 可以独立计算。
- [ ] 能够识别训练集过拟合现象。
- [ ] 可以保存和重新加载模型。
- [ ] 模型参数确实随优化步骤发生变化。

不要求达到固定的文本生成质量。

### 6.3 交付物

`dataset.py`、`train.py`、`checkpoint.py`、`tests/test_training.py`

训练记录至少包括 Step、Training Loss、Validation Loss、Learning Rate 和模型配置。

---

# 阶段五：自回归文本生成

## 7. 阶段目标

让训练完成的模型可以根据 Prompt 连续生成新的 Token。

### 7.1 功能需求

**REQ-5.1：基础生成**

实现接口：

`generate(prompt, max_new_tokens)`

基本流程：

1. 将 Prompt 编码为 Token ID。
2. 输入模型进行 Forward。
3. 取最后一个位置的 Logits。
4. 选择下一个 Token。
5. 追加到当前 Token 序列。
6. 重复，直到达到生成长度上限。
7. 将 Token ID 解码为文本。

如果生成序列超过 Context Length，只保留最近的 Context Length 个 Token 作为模型输入。

**REQ-5.2：生成策略**

至少支持：

- Greedy Decoding
- Temperature Sampling
- Top-k Sampling

要求支持设置随机种子，便于复现实验。

**REQ-5.3：推理模式**

要求使用 `model.eval()` 和 `torch.no_grad()` 或 `torch.inference_mode()`。

不得在推理阶段执行反向传播。

### 7.2 验收标准

- [ ] 能够使用训练完成的 Checkpoint 生成文本。
- [ ] 生成长度符合 `max_new_tokens` 限制。
- [ ] Greedy Decoding 在相同输入和模型下可复现。
- [ ] Temperature 和 Top-k 能够实际影响采样过程。
- [ ] 超过上下文长度时不会产生维度错误。
- [ ] 推理阶段不生成梯度。
- [ ] 对包含词表外字符的 Prompt 有明确处理策略。

### 7.3 交付物

`generate.py`、`tests/test_generation.py`

最终交付一个简单命令行程序，能够输入 Prompt 并显示生成结果。

**本阶段完成，即视为 Mini-GPT v0.1 核心目标达成。**

---

# 扩展阶段：Byte-level BPE 与模型实验

## 8. v0.2 可选需求

核心 v0.1 验收完成后，再进入扩展阶段。

### 8.1 自行实现 Byte-level BPE

实现：

- 从基础字节集合初始化。
- 统计相邻 Token 对。
- 按频率选择合并对象。
- 保存 Merge Rules 和 Rank。
- 编码、解码。
- 文件序列化。

验证任意有效 UTF-8 文本的无损往返转换，并测试不属于常见语言的符号。

对于任意原始二进制数据的支持，可以单独设计 bytes 接口。

### 8.2 对比不同 Tokenizer

保持训练语料和模型架构尽量一致，比较：

| 指标 | Character-level | Byte-level BPE |
|---|---|---|
| Vocabulary Size | 记录 | 记录 |
| Token 序列长度 | 记录 | 记录 |
| 训练速度 | 记录 | 记录 |
| Validation Loss | 记录 | 记录 |
| 生成文本 | 观察 | 观察 |

注意，两个 Tokenizer 产生的 Token 定义不同，不能仅凭每 Token 的 Loss 直接判断哪个模型更好。可以进一步使用按字符或字节归一化的指标比较。

更换 Tokenizer 后，通常需要重新初始化并训练对应模型，不直接复用原模型的 Token Embedding。

### 8.3 Transformer 消融实验

使用同一训练与验证方案，分别测试：

- 关闭 Attention 更新分支。
- 关闭 MLP 更新分支。
- 移除显式位置 Embedding。
- 改变 Attention Head 数量。
- 改变 Transformer 层数。

分别记录训练结果、验证结果和观察到的差异。

Causal Mask 的移除实验应当作为**信息泄漏诊断实验**单独处理，不能把其训练 Loss 与正常自回归模型的 Loss 直接视为公平比较。

### 8.4 KV Cache

在基础模型正确性得到验证后，实现 KV Cache，并比较：

- Cache 开启与关闭的生成结果是否一致。
- 不同上下文长度下的推理耗时。
- Cache 对显存或内存占用的影响。

---

# 9. 项目验收与开发纪律

每个阶段结束时，必须产生以下交付内容：

1. **实现代码**：仅包含当前阶段需要的功能。
2. **单元测试**：覆盖核心行为、维度和主要边界条件。
3. **运行记录**：包含测试结果和关键输出。
4. **阶段说明**：说明输入输出契约、主要设计选择和当前限制。
5. **验收结论**：PASS / FAIL / BLOCKED。

未通过当前阶段的核心验收，不进入下一阶段。

不为了提前优化性能而引入额外的框架或复杂抽象。

# 10. 最终验收定义

当 Mini-GPT v0.1 满足以下条件时，项目正式完成：

- [ ] 从本地文本建立 Tokenizer。
- [ ] 独立实现 Embedding、QKV、Self-Attention、MLP 和 Transformer Block。
- [ ] 不依赖预训练模型。
- [ ] 能够完整执行前向传播和反向传播。
- [ ] 能够在小样本上成功过拟合。
- [ ] 能够使用正常语料训练并观察 Loss。
- [ ] 能够保存和恢复模型。
- [ ] 能够根据 Prompt 自回归生成文本。
- [ ] 所有核心模块有自动化测试。
- [ ] 可以独立解释每个主要组件的计算过程。

**项目最终成果：一个从随机初始化开始训练、结构可解释、可以运行和测试的教学型 GPT。**
