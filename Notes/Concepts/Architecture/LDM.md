---
type: concept
aliases: [Latent Diffusion Model, 隐扩散模型, Stable Diffusion]
---

# LDM

## 定义
Latent Diffusion Model：在 VAE 编码的低维隐空间中运行扩散过程，而非直接在像素空间，大幅降低计算成本同时保持生成质量。

## 数学形式
$$z = \mathcal{E}(x), \quad \hat{z} = \text{DM}(z), \quad \hat{x} = \mathcal{D}(\hat{z})$$

## 核心要点
1. VAE 将高分辨率图像压缩到 4-16× 小的隐空间（f=4/8/16）
2. 扩散过程完全在隐空间进行，节省大量计算
3. 条件信息（文本、图像）通过 cross-attention 注入 UNet/DiT
4. Stable Diffusion 系列的基础架构

## 代表工作
- Rombach et al. 2022，High-Resolution Image Synthesis with Latent Diffusion Models
- [[FLUX]]、[[SD3]] 使用 [[DiT]] + Flow Matching 替换 UNet + DDPM

## 相关概念
- [[DDPM]]
- [[DiT]]
- [[Flow Matching]]
- [[Diffusion Model]]
