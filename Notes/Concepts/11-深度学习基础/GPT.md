---
type: concept
aliases: [Generative Pre-trained Transformer, GPT-2, GPT-3, GPT-4, 因果语言模型, CLM]
---

# GPT

## 定义

GPT（Generative Pre-trained Transformer）是基于**因果自注意力** Transformer 的自回归语言模型，通过**下一 token 预测**（next-token prediction）在大规模语料上预训练，是现代大语言模型的基础架构范式。

## 数学形式

$$
\mathcal{L}_{\text{GPT}} = -\sum_{t} \log p_\theta(X_t \mid X_{1:t-1})
$$

## 核心要点

1. **因果掩码**：自注意力只能看到当前及之前的 token，保证自回归生成的因果性
2. **预训练-微调范式**：大规模无监督预训练 + 下游任务微调
3. **涌现能力**：随参数量和数据量扩展，出现少样本学习、推理等涌现能力
4. **隐状态无约束**：中间隐状态 $h_t$ 仅被 token 预测目标隐式约束，不保证信念状态性质

## 代表工作

- OpenAI GPT 系列（GPT-2, GPT-3, GPT-4）
- LLaMA、Mistral 等开源变体
- [[NextLat]]: 以 GPT 为 Backbone，通过潜态监督增强其世界建模能力

## 相关概念

- [[自回归Transformer]]
- [[Transformer]]
- [[信念状态]]
- [[多头预测 (MTP)]]
