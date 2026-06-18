---
type: concept
aliases: [跨视图注意力, Cross-View Self-Attention]
---

# Cross-View Attention

## 定义

跨视图注意力是一种多视图生成模型中的注意力机制，允许某一相机视角的 token 通过注意力机制 attend 到其他所有视角的 key/value，实现视角间的显式信息交换，从而建模多视角一致性约束。

## 数学形式

$$
\hat{Z}^v = Z^v + \text{gate} \cdot \text{softmax}\!\left(\frac{Q^v [K^1; \cdots; K^V]^\top}{\sqrt{d}}\right)[V^1; \cdots; V^V]
$$

其中 $Q^v$ 仅来自当前视角 $v$，而 $K, V$ 来自所有视角的拼接。

## 核心要点

1. **视角间通信**: 每个视角可以"看到"其他视角的信息，解决 flat-concat 无通信通路的问题
2. **几何偏置**: 结合 [[Geometric RoPE]] 后，注意力权重受相机几何约束，优先关注几何对应区域
3. **插入位置**: 通常在预训练 DiT 的选定层中插入，配合 [[AdaLN-Zero]] 零初始化
4. **计算复杂度**: 随视角数线性增长（$V \times H \times W$ token）

## 代表工作

- [[PAIWorld]]: 提出 Geometry-Aware Cross-View Attention，结合 Geo-RoPE 实现几何感知的跨视图注意力

## 相关概念

- [[Geometric RoPE]]
- [[AdaLN-Zero]]
- [[RoPE]]
- [[Diffusion Transformer]]
