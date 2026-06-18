---
type: concept
aliases: [几何感知跨视图注意力, GACVA]
---

# Geometry-Aware Cross-View Attention

## 定义

Geometry-Aware Cross-View Attention 是 [[PAIWorld]] 提出的核心组件，在 [[Cross-View Attention]] 基础上引入 [[Geometric RoPE]]，使跨视图注意力的 query/key 携带相机几何信息，从而实现几何偏置的视角间信息交换。

## 数学形式

$$
\hat{Z}^v_t = Z^v_t + \text{gate} \cdot \text{softmax}\!\left(\frac{\tilde{Q}^v_t\,[\tilde{K}^1_t;\cdots;\tilde{K}^V_t]^\top}{\sqrt{d}}\right) [V^1_t;\cdots;V^V_t]
$$

其中 $\tilde{Q}^v_t, \tilde{K}^v_t$ 为经 Geo-RoPE 旋转后的 query 和 key，gate 由 [[AdaLN-Zero]] 零初始化。

## 核心要点

1. **几何偏置注意力**: Geo-RoPE 使得空间上几何对应的 token 对之间注意力权重更高
2. **两种注意力模式**: 选定层做跨视图注意力；周期性地做空间维度联合注意力（$V \cdot H \cdot W$ token）
3. **预训练保护**: AdaLN-Zero gate 初始化为 0，不破坏基础模型 Cosmos-Predict2.5 的预训练权重
4. **协同 3D-REPA**: 单独使用时会退化为纹理复制，需配合 [[Latent 3D-REPA]] 提供几何监督

## 代表工作

- [[PAIWorld]]: 提出此方法，实现机器人操作世界模型的多视图 3D 一致性生成

## 相关概念

- [[Cross-View Attention]]
- [[Geometric RoPE]]
- [[AdaLN-Zero]]
- [[Latent 3D-REPA]]
