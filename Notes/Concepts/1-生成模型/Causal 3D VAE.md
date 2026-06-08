---
type: concept
aliases: [因果三维变分自编码器, Causal Temporal VAE, 3D VAE]
---

# Causal 3D VAE

## 定义
一种视频/音频专用的变分自编码器，在时序维度上使用因果（单向）卷积，对视频帧序列做时序压缩，保留帧间运动语义，尤其针对快速运动目标做了优化。

## 数学形式

$$
z = \mathcal{E}(x_0, x_1, \ldots, x_T) \in \mathbb{R}^{T' \times H' \times W' \times C}
$$

其中 $T' < T$（时序下采样），$\mathcal{E}$ 采用因果卷积确保 $z_t$ 只依赖 $x_{\leq t}$。

## 核心要点
1. **因果性**: 使用单向时序卷积，确保 latent code 不泄露未来帧信息
2. **时序压缩**: 将原始视频帧压缩到更低的时间分辨率，大幅减少 token 数量
3. **运动保持**: 专门优化 latent 空间中的运动表示，对快速运动物体鲁棒
4. **多模态支持**: 可同时处理视频和音频的时序压缩

## 代表工作
- [[Cosmos3]]: 使用 Causal 3D VAE 作为视频和音频的核心 tokenizer，生成连续 DM token

## 相关概念
- [[变分自编码器]]
- [[扩散世界模型]]
- [[视频基础模型]]
