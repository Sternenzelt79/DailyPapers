---
type: concept
aliases: [Latent 3D REPA, 潜在3D表示对齐, 3D-REPA]
---

# Latent 3D-REPA

## 定义

Latent 3D-REPA 是基于 [[REPA]] 框架的改进方法，使用冻结的 3D 感知基础模型（[[Depth Anything 3]]）作为教师，通过 **token 关系蒸馏**（而非特征值蒸馏）为扩散 DiT 提供几何监督信号，解决特征空间对齐问题。

## 数学形式

**相似度矩阵**（token 关系，避免直接对齐特征值）：

$$
S(F)_{i,a} = \frac{f_i^\top f_a}{\|f_i\| \cdot \|f_a\|}, \quad a \in \mathcal{A}
$$

**蒸馏损失**：

$$
\mathcal{L}_{\text{REPA}} = \underbrace{\text{SmoothL1}(S^{\text{DiT}}_{\text{intra}}, S^{\text{DA3}}_{\text{intra}})}_{\mathcal{L}_{\text{spatial}}} + \underbrace{\text{SmoothL1}(S^{\text{DiT}}_{\text{inter}}, S^{\text{DA3}}_{\text{inter}})}_{\mathcal{L}_{\text{temporal}}}
$$

## 核心要点

1. **关系蒸馏而非特征蒸馏**: 对齐 token 之间的余弦相似度矩阵而非特征值，绕过 DiT 与深度估计器特征空间不兼容的问题
2. **两级监督**: 空间项（帧内 token 关系）+ 时间项（跨帧 token 关系），同时约束几何和时序一致性
3. **Anchor 采样降复杂度**: 从 $N$ 个 token 中采样 $M$ 个 anchor，复杂度从 $O(N^2)$ 降至 $O(M \cdot K)$
4. **超加性协同**: 单独使用仅改善 MEt3R 0.72，配合 [[Geometry-Aware Cross-View Attention]] 时协同改善 2.64

## 代表工作

- [[REPA]]: 原始表示对齐框架
- [[PAIWorld]]: 提出 Latent 3D-REPA 将 REPA 扩展到 3D 几何对齐场景

## 相关概念

- [[REPA]]
- [[Depth Anything 3]]
- [[Token Relation Distillation]]
- [[Geometry-Aware Cross-View Attention]]
