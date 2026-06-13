---
type: concept
aliases: [视频变分自编码器, Video Variational Autoencoder, Video VAE]
---

# Video VAE

## 定义
针对视频序列设计的变分自编码器，将时空维度的 RGB 视频压缩到低维潜空间，再通过解码器重建。相比图像 VAE，Video VAE 在时间维度上引入卷积或注意力机制以捕捉帧间一致性。

## 数学形式
编码：$z_v = \mathcal{E}(I_{0:T}) \in \mathbb{R}^{C \times L \times H' \times W'}$

解码：$\hat{I}_{0:T} = \mathcal{D}(z_v)$

其中 $L \ll T$，$H' \ll H$，$W' \ll W$，表示时空压缩。

## 核心要点
1. **时空压缩**: 将视频从像素空间压缩到低维潜空间，使扩散/流匹配训练计算可行
2. **冻结使用**: 在 WAM 类方法（[[FastWAM]]、[[MaskWAM]]）中，预训练 Video VAE 通常冻结参数，仅用作特征提取器
3. **跨模态复用**: MaskWAM 证明 Video VAE 可同时编码 RGB 帧和三通道 Mask 图像，共享潜空间实现通道级融合

## 代表工作
- [[FastWAM]]: 率先在 WAM 框架中使用冻结 Video VAE 编码视觉观测
- [[MaskWAM]]: 扩展 Video VAE 同时处理 RGB 和 Mask，通道拼接后送入 [[Diffusion Transformer]]

## 相关概念
- [[VAE]]
- [[潜扩散模型]]
- [[Diffusion Transformer]]
- [[World Action Model]]
