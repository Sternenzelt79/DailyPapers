---
type: concept
aliases: [Vision Transformer, 视觉Transformer]
---

# ViT

## 定义
Vision Transformer，将图像切分为固定大小的 patch 并展开为序列，直接应用标准 Transformer 自注意力机制进行视觉理解的模型架构。

## 数学形式
Patch embedding：
$$z_0 = [x_{\text{cls}}; x_p^1 E; x_p^2 E; \ldots; x_p^N E] + E_{\text{pos}}$$

其中 $E \in \mathbb{R}^{(P^2 C) \times D}$ 为线性投影矩阵，$P$ 为 patch 大小。

## 核心要点
1. 无卷积归纳偏置，需要大量数据预训练
2. 全局自注意力捕获长程依赖，适合语义级理解
3. 是 VLA 模型（如 GR00T、π0）视觉 backbone 的主流选择

## 代表工作
- Dosovitskiy et al. (2021): An Image is Worth 16x16 Words
- [[EquiVLA]]: 以冻结 ViT 为基础提取等变视觉特征

## 相关概念
- [[DiT]]
- [[VLA]]
