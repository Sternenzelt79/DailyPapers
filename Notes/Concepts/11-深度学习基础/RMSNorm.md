---
type: concept
aliases: [RMSNorm, Root Mean Square Normalization, 均方根归一化]
---

# RMSNorm

## 定义
Layer Normalization 的简化变体，只对激活值的均方根（RMS）进行归一化，去掉均值中心化步骤，计算更高效且效果相当。

## 数学形式
$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \gamma
$$
$$
\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}
$$

其中 $\gamma$ 为可学习的缩放参数。

## 核心要点
1. 相比 LayerNorm 去掉了均值计算，速度约快 10-20%
2. 在 LLaMA、GPT-NeoX 等主流大语言模型中广泛使用
3. 训练稳定性与 LayerNorm 相当

## 代表工作
- [[WEAVER]]: 用于 32 层潜动力学 Transformer 的归一化

## 相关概念
- [[Transformer]]
- [[QKNorm]]
- [[SwiGLU]]
