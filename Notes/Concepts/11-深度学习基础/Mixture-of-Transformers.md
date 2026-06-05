---
type: concept
aliases: [MoT, Mixture of Transformers, 混合变换器]
---

# Mixture-of-Transformers

## 定义
一种 Transformer 架构变体，在同一前向传播中使用多组独立的参数集（类似 MoE 的"专家"概念），对不同类型的 token 分别应用各自的参数，同时允许 token 间的跨组注意力交互。

## 数学形式

对于 token 序列 $X = [X_1, X_2]$（两个子序列），MoT 层计算：

$$
H_1 = \text{Attention}_{\theta_1}(X_1, [X_1; X_2]) + \text{FFN}_{\theta_1}(X_1)
$$

$$
H_2 = \text{Attention}_{\theta_2}(X_2, [X_1; X_2]) + \text{FFN}_{\theta_2}(X_2)
$$

其中 $\theta_1 \neq \theta_2$ 为独立参数，但注意力的 key/value 来自合并序列（joint attention）。

## 核心要点
1. **参数隔离**: 不同模态/类型的 token 使用独立参数，避免相互干扰
2. **Joint Attention**: token 间通过共享注意力实现信息交互
3. **灵活组合**: 可同时处理离散 token（AR）和连续 token（扩散）
4. **高效扩展**: 相比 MoE 无需 routing，计算更可预测

## 代表工作
- [[Cosmos3]]: 核心架构，AR Tower（离散推理）和 DM Tower（连续生成）使用独立参数但通过 joint attention 交互

## 相关概念
- [[Two-Tower MoT]]
- [[自回归变换器]]
- [[Diffusion Transformer]]
