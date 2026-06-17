---
type: concept
aliases: [寄存器Token, 寄存器标记, Register]
---

# Register Tokens

## 定义

附加在 Vision Transformer 输入序列末尾（或插入中间层）的可学习全局 Token，不对应任何图像 patch，作为"全局信息汇聚槽"或"几何查询探针"，使模型能够将全局/跨区域信息从注意力矩阵的 patch Token 中解耦出来。

## 数学形式

$$
\mathbf{R}^{(\ell+1)} = \text{Attention}\!\left(Q=\mathbf{R}^{(\ell)},\; K{,}V=[\mathbf{R}^{(\ell)},\; \mathbf{Z}^{(\ell)}]\right)
$$

寄存器以自注意力 + 对图像 Token $\mathbf{Z}$ 的交叉注意力方式更新，提取所需信息。

## 核心要点

1. **原始用途（DINOv2 Register）**: 减少 ViT 中的"artifact"——高显著性 patch 吸引全局注意力导致的注意力噪声问题
2. **扩展用途（WAM4D）**: 作为几何查询探针，在训练时蒸馏深度几何先验，推理时整体移除，实现零开销部署
3. **空间对应**: 可为每个寄存器赋予空间坐标（RoPE 编码），使其对应特定图像区域
4. **模态隔离**: 通过注意力掩码限制寄存器只访问特定 Token 组，防止信息泄漏

## 代表工作

- [[WAM4D]]: 空间寄存器 Token 用于几何知识蒸馏，960 个 Token（3 视角 × 8 帧 × 12×10 网格）
- DINOv2 Registers（Darcet et al., 2023）: 原始 Register Token 提案，解决 ViT artifact 问题

## 相关概念

- [[Knowledge Distillation]]
- [[Mixture-of-Transformers]]
- [[Depth Anything 3]]
