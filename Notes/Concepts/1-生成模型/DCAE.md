---
type: concept
aliases: [Deep Compression AutoEncoder, 深度压缩自编码器]
---

# DCAE

## 定义
Deep Compression AutoEncoder，一种高压缩比的视觉自编码器，用于将图像/视频压缩到极低维度的潜在空间，同时保持生成质量，是 SANA 系列视频扩散模型的核心组件。

## 数学形式
$$z = \mathcal{E}(x), \quad x' = \mathcal{D}(z), \quad z \in \mathbb{R}^{H/r \times W/r \times C}$$

其中压缩比 $r$ 远大于标准 VAE（DCAE 典型 $r=32$，标准 VAE $r=8$）。

## 核心要点
1. 比标准 [[VAE]] 压缩率更高（4× 以上），大幅减少 DiT token 数量
2. 压缩比提升直接降低 [[DiT]] 的计算量（token 数量平方级别减少）
3. 专为视频生成场景设计，支持时序连贯的潜在空间表示

## 代表工作
- [[SANA-Video]]: 使用 DCAE 实现高效长视频生成，720×1280 分辨率可在 RTX 5090 部署

## 相关概念
- [[VAE]]
- [[DiT]]
- [[潜扩散模型]]
