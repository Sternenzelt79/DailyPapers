---
type: concept
aliases: [Large Language Model, 大语言模型, 大模型]
---

# LLM（大语言模型）

## 定义
Large Language Model（LLM）是基于 Transformer 架构、在大规模文本语料上预训练的语言模型，具备强大的语言理解、生成和推理能力，参数量通常在数十亿以上。

## 数学形式

$$
P(x_{t+1} \mid x_1, \ldots, x_t) = \mathrm{softmax}\!\left(W_h \cdot h_t\right)
$$

其中 $h_t$ 为 Transformer 在位置 $t$ 的隐状态，$W_h$ 为输出投影矩阵。

## 核心要点
1. **预训练 + 微调范式**：大规模无监督预训练 + 下游任务 SFT / RLHF
2. **涌现能力**：随参数规模增大出现零样本推理、上下文学习等能力
3. **指令跟随**：RLHF / DPO 对齐训练使模型能理解自然语言指令
4. **在 VLA 中的作用**：作为语义编码器（如 InternVL、PaliGemma）或数据生成工具（如 Qwen3 生成指令变体）

## 代表工作
- [[RoVLA]]: 使用 [[Qwen3|Qwen3-8B]] 离线生成约 15 条指令变体/轨迹，实现 Instructional Consistency

## 相关概念
- [[VLM]]
- [[Qwen3]]
- [[InternVL]]
- [[SFT]]
- [[RLHF]]
