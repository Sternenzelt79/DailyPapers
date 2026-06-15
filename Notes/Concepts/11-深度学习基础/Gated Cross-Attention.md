---
type: concept
aliases: [门控交叉注意力, Gated Cross Attention, GCA]
---

# Gated Cross-Attention

## 定义

Gated Cross-Attention 是一种带可学习门控标量的交叉注意力机制，通过 Sigmoid 门控调制外部特征对主特征流的影响强度，实现软性条件融合。

## 数学形式

$$
\mathbf{z}_{\text{guided}} = \mathbf{z} + \sigma(g) \cdot \mathrm{CA}(Q = \mathrm{LN}(\mathbf{z}),\; K = V = \mathbf{h}_{\text{ext}})
$$

- $g$: 可学习标量门控，初始化为接近 0（保证训练初期主路径主导）
- $\sigma(\cdot)$: Sigmoid 函数
- $\mathrm{CA}$: 标准多头交叉注意力

## 核心要点

1. **软性融合**: 门控允许模型学习"何时"以及"多少"注入外部信息
2. **训练稳定性**: 门控初始化为 0 时，训练初期等同于不加外部特征，渐进融合
3. **参数高效适配**: 冻结主干后，只需训练门控和交叉注意力参数即可适配外部特征

## 代表工作

- [[mu0]]: 冻结预训练 μ₀ 后，用 Gated Cross-Attention 将中间 trace 去噪特征注入 VLM 特征，训练动作专家头

## 相关概念

- [[Conditional Flow Matching]]
- [[VLM]]
