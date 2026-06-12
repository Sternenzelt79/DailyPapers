---
type: concept
aliases: [QKNorm, QK Normalization, Query-Key Normalization]
---

# QKNorm

## 定义
在 Transformer 注意力计算前，对 Query 和 Key 向量分别做 L2 归一化，防止注意力 logit 数值爆炸，提升大模型训练稳定性。

## 数学形式
$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{\hat{Q}\hat{K}^\top}{\tau}\right)V
$$
$$
\hat{Q}_i = \frac{Q_i}{\|Q_i\|_2}, \quad \hat{K}_j = \frac{K_j}{\|K_j\|_2}
$$

其中 $\tau$ 为可学习的温度参数。

## 核心要点
1. 防止 Q/K 内积随维度增大或网络加深而发散
2. 对超大模型（如 ViT-22B、GPT-4 级别）尤为重要
3. 通常与 RMSNorm 搭配使用

## 代表工作
- [[WEAVER]]: 作为 32 层动力学 Transformer 的标准组件

## 相关概念
- [[RMSNorm]]
- [[Transformer]]
- [[2D Transformer]]
