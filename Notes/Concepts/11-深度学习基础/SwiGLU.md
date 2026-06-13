---
type: concept
aliases: [SwiGLU, Swish-Gated Linear Unit]
---

# SwiGLU

## 定义
GLU（Gated Linear Unit）的变体，将 Swish 激活函数与门控机制结合，用于替代 Transformer FFN 层中的 ReLU，在参数量相近的情况下提供更强的表达能力。

## 数学形式
$$
\text{SwiGLU}(x, W, V, b, c) = \text{Swish}(xW + b) \odot (xV + c)
$$
$$
\text{Swish}(x) = x \cdot \sigma(x)
$$

其中 $\sigma$ 为 Sigmoid 函数，$\odot$ 为逐元素乘法。

## 核心要点
1. 门控机制允许网络选择性地通过信息，增强非线性建模能力
2. 通常需要将 FFN 隐藏维度缩小到 $\frac{2}{3}$ 以保持参数量与 ReLU FFN 相当
3. 在 PaLM、LLaMA 等主流 LLM 中被广泛采用

## 代表工作
- [[WEAVER]]: 作为 32 层潜动力学 Transformer FFN 的激活函数

## 相关概念
- [[Transformer]]
- [[RMSNorm]]
- [[2D Transformer]]
