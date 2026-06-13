---
type: concept
aliases: [Stable Diffusion 3, SD3]
---

# Stable Diffusion 3

## 定义
Stability AI 发布的第三代扩散图像生成模型，采用 Flow Matching 训练目标和 Multimodal Diffusion Transformer（MMDiT）架构，并配备高质量的 VAE 编码器/解码器。

## 核心要点
1. **Flow Matching 训练**: 用整流流取代传统 DDPM 噪声目标，生成更直接、高效
2. **MMDiT 架构**: 联合处理文本和图像 token，实现强文本对齐
3. **高质量 VAE**: SD3 的 VAE 编码器/解码器将图像压缩到 16 通道潜空间，重建质量优于 SD1.x/SD2.x
4. **预训练表示迁移**: SD3 VAE 常被迁移到其他任务（如机器人世界模型）作为通用视觉特征提取器

## 代表工作
- [[WEAVER]]: 直接复用 SD3 的 VAE 编码器 $\mathcal{E}_\psi$ 和解码器 $\mathcal{D}_\eta$ 作为感知基础，避免从头学习视觉表示

## 相关概念
- [[VAE]]
- [[Flow Matching]]
- [[Rectified Flow]]
