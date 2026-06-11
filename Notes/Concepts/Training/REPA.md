---
type: concept
aliases: [Representation Alignment, 表征对齐]
---

# REPA (Representation Alignment)

## 定义
在 latent diffusion 训练中，将预训练视觉模型（如 DINO、SigLIP）的表征注入扩散模型生成器，加速收敛并提升生成质量的对齐技术。

## 数学形式
$$\mathcal{L}_\text{REPA} = \mathcal{L}_\text{diffusion} + \lambda \cdot \text{sim}(f_\text{VFM}(x), h_\theta(z_t, t))$$

其中 $f_\text{VFM}$ 是冻结的视觉基础模型，$h_\theta$ 是扩散模型的中间层表征，$\text{sim}$ 为余弦相似度或 MSE。

## 核心要点
1. 扩散模型本身是图像重建目标，不能自然地学习到视觉语义结构；REPA 通过蒸馏外部视觉模型的表征来注入语义先验
2. 在生成器（DiT 等）侧操作，区别于在 tokenizer 侧操作的 [[PAE]]
3. 使用 DINO、SigLIP2 等强视觉编码器作为表征来源

## 代表工作
- 原始 REPA 论文：Regularizing Denoising Networks with Pretrained Representations
- [[PAE]]：在 tokenizer 侧类似地引入表征对齐
- [[AGRA]]：将 REPA 对齐技术重新定向为 WAM 的动作基础接口正则化器（而非改善生成质量）

## 相关概念
- [[Diffusion Model]]
- [[PAE]]
- [[DINO-WM]]
- [[SigLIP2]]
