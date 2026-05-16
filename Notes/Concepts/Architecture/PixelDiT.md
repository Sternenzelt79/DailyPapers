---
type: concept
aliases: [PixelDiT, Pixel-space Diffusion Transformer]
---

# PixelDiT

## 定义
PixelDiT：将 [[DiT]]（Diffusion Transformer）架构直接应用于像素空间生成，不经过 VAE 隐空间压缩，以获得更高保真度的图像生成结果。

## 数学形式
与 [[DiT]] 相同，但输入输出均为像素 patch：
$$x \in \mathbb{R}^{H \times W \times C}, \quad \text{patches} \in \mathbb{R}^{N \times (p^2 C)}$$

## 核心要点
1. 省去 VAE 编解码步骤，避免重建误差
2. 计算代价远高于 LDM（像素维度高），对硬件要求极高
3. 与 [[PixelFlow]] 类似，是 "pixel-space 生成复兴" 趋势的一部分

## 代表工作
- [[L2P]] 等工作尝试用 LDM 知识迁移降低 PixelDiT 训练成本

## 相关概念
- [[DiT]]
- [[PixelFlow]]
- [[LDM]]
