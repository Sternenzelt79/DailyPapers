---
concept: Cross-Attention
aliases: [Cross Attention, CA]
category: Architecture
tags: [transformer, attention]
created: 2026-05-09
---

# Cross-Attention

## 定义

跨注意力：query 来自一个序列、key/value 来自另一个序列的注意力机制，用于条件化生成或模态融合。

## 标准形式

$$
\text{CA}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
$$

## 在 [[VLA]] 中的应用

[[Action Expert]] 通过 CA 读取 [[VLM Backbone|VLM]] 各层的 KV，避免重算 VLM。例如 [[MolmoAct2]] 用投影矩阵：

$$
\tilde{K}_\ell = \text{reshape}(P_K K^{vlm}_\ell), \quad \tilde{V}_\ell = \text{reshape}(P_V V^{vlm}_\ell)
$$

## 代表工作

- [[Transformer]] (2017): 原始提出。
- [[MolmoAct2]] (2026): KV 投影 + 三处门控。
- [[Pi0]] / [[Pi05]]: VLM-Action 解耦的 CA 用法。

## 相关概念

- [[Self-Attention]]
- [[Action Expert]]
