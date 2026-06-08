---
type: concept
aliases: [Patch Embedder, Patch Tokenization, 补丁嵌入]
---

# Patch Embedding（补丁嵌入）

## 定义
将图像或视频帧切割为固定大小的补丁（patch），再通过线性投影或卷积映射为固定维度的向量序列，作为 Transformer 的输入 token。

## 数学形式

$$
z_i = W_e \cdot \text{flatten}(x_i) + b_e
$$

其中 $x_i$ 为第 $i$ 个 patch 的像素值，$W_e \in \mathbb{R}^{d \times (P^2 C)}$ 为线性投影矩阵，$z_i \in \mathbb{R}^d$ 为输出嵌入向量。

## 核心要点

1. **标准用法**: 将 $H \times W$ 图像切为 $N = HW/P^2$ 个 $P \times P$ 的补丁，线性映射为 $d$ 维向量
2. **视频扩展**: 在时序维度增加 temporal patch size，处理 3D 时空补丁
3. **多源融合**: 可为不同条件（如 RGB 帧 + 骨骼图）使用独立的 patch embedder，分别编码再在潜空间融合（见 [[OSCAR]]）

## 代表工作

- [[ViT]]: Patch Embedding 的奠基工作（将图像视为补丁序列）
- [[Diffusion Transformer]]: DiT 中使用 patch embedding 将含噪图像转为 token 序列
- [[OSCAR]]: 骨骼帧和视频帧各用独立 patch embedder，融合后送入 DiT 去噪

## 相关概念

- [[Diffusion Transformer]]: 主要使用场景
- [[Transformer]]: 上层架构
