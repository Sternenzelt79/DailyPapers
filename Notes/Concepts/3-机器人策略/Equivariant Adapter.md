---
type: concept
aliases: [等变适配器, EquivariantAdapter, 等变门控融合]
---

# Equivariant Adapter

## 定义

Equivariant Adapter 是 EquiVLA 中用于融合等变几何特征流和不变语义特征流的可学习模块，通过不变门控（invariant gate）在两个信息流之间自适应加权，同时保持输出的等变性。

## 数学形式

$$
\mathbf{z}^{out}_{eq} = \alpha^{reg} \odot W_s(s^{inv} \otimes \mathbf{1}_{|G|}) + (1 - \alpha^{reg}) \odot W_g(\tilde{\mathbf{z}}^{eq})
$$

门控权重：

$$
\alpha = \sigma(W_{gate}[s^{lang};\ s^{vis};\ \bar{\mathbf{z}}^{eq}]), \quad \alpha^{reg} = \alpha \otimes \mathbf{1}_{|G|}
$$

## 核心要点

1. **等变保持**: 门控 $\alpha$ 由不变量（$s^{lang}, s^{vis}, \bar{\mathbf{z}}^{eq}$）计算，保证输出 $\mathbf{z}^{out}_{eq}$ 仍为等变 token
2. **信息互补**: 不变语义流（来自冻结 VLM 上下文）提供高层语义，等变几何流提供空间关系
3. **轻量可学习**: 仅 $W_s, W_g, W_{gate}$ 三个线性层需要训练，冻结 VLM 权重不变

## 代表工作

- [[EquiVLA]]: Equivariant Adapter 是 EquiPerceptor 的关键组件，连接等变视觉 token 与冻结 VLM 上下文

## 相关概念

- [[Frame Averaging]]
- [[SO(2) Equivariance]]
- [[VLA（Vision-Language-Action Model）]]
