---
type: concept
aliases: [Prior-Aligned Autoencoder, 先验对齐自编码器]
---

# PAE (Prior-Aligned Autoencoder)

## 定义
一种为 latent diffusion 设计的自编码器，通过对齐 latent 分布与标准高斯先验，使潜在空间对扩散模型更友好，从而提升生成质量和训练效率。

## 数学形式
$$\mathcal{L} = \mathcal{L}_\text{recon} + \lambda_\text{align} \mathcal{L}_\text{prior}$$

其中 $\mathcal{L}_\text{prior}$ 通过 SSC、MCR、SCR、SSR 等正则化项约束 latent 分布接近标准高斯 $\mathcal{N}(0, I)$。

## 核心要点
1. 现有 tokenizer 只优化重建质量，忽视了 latent 几何结构对 diffusion 收敛的影响
2. PAE 在编码器端注入视觉表征先验（如 [[SigLIP2]]、[[DINO-WM]]），使 latent space 同时满足重建和扩散两个目标
3. 与 [[REPA]] 类似但更系统——REPA 在生成器侧注入表征先验，PAE 在 tokenizer 侧设计先验对齐
4. 在 [[ImageNet]] 基准上验证，配合 [[DiT]] 框架效果显著

## 代表工作
- [[What Matters for Diffusion-Friendly Latent Manifold]]: 提出 PAE 的原始论文（2026-05-12 推荐）

## 相关概念
- [[Diffusion Model]]
- [[REPA]]
- [[SigLIP2]]
- [[DiT]]
