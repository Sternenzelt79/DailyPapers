---
type: concept
aliases: [SigLIP 2, Sigmoid Loss for Image-Language Pretraining v2]
---

# SigLIP2

## 定义
Google 开发的视觉-语言对比预训练模型第二代，使用 sigmoid 损失替代 softmax，支持更大 batch 和更高效的训练，广泛用作视觉表征骨干网络。

## 数学形式
$$\mathcal{L}_\text{SigLIP} = -\frac{1}{n^2} \sum_{i,j} \log \sigma\left(z_{ij} \cdot (y_{ij} \cdot 2 - 1)\right)$$

其中 $z_{ij} = \tau \cdot \langle v_i, t_j \rangle$，$y_{ij} \in \{0,1\}$ 表示图文是否匹配，$\sigma$ 为 sigmoid 函数。

## 核心要点
1. 相比 CLIP 的 softmax 对比损失，sigmoid 损失不需要跨设备全局 normalization，天然支持极大 batch
2. SigLIP2 在 SigLIP 基础上改进了多粒度对比学习和数据配方
3. 常被用作 latent diffusion 的视觉表征先验来源（如 [[REPA]]、[[PAE]]）
4. 在 VLA 和多模态模型中也作为视觉 backbone 广泛使用

## 代表工作
- SigLIP (2023): 原始论文，提出 sigmoid 对比损失
- SigLIP2 (2024): 扩展版，更强的视觉-语言对齐

## 相关概念
- [[DINO-WM]]
- [[PAE]]
- [[REPA]]
- [[VLM]]
