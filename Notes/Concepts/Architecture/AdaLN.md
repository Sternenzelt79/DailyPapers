---
type: concept
aliases: [Adaptive Layer Normalization]
---

# AdaLN (Adaptive Layer Normalization)

## 定义
将条件 $c$（如时间步、动作、类别）通过 MLP 映射为 LayerNorm 的 scale/shift 参数，从而把条件注入 Transformer 块。

## 数学形式
$$
\text{AdaLN}(x, c) = \gamma(c) \odot \frac{x - \mu}{\sigma} + \beta(c)
$$

## 核心要点
1. 比 cross-attention 注入更轻量。
2. 在 [[DiT]]、[[LeWM]] 等扩散/世界模型中作为标准条件机制。
3. 变体 AdaLN-Zero：初始化 $\gamma=0$ 让残差从 identity 起步。

## 代表工作
- [[DiT]]：扩散 Transformer 中 AdaLN-Zero
- [[LeWM]]：predictor 用 AdaLN 注入动作

## 相关概念
- [[Layer Normalization]]
- [[Vision Transformer]]
