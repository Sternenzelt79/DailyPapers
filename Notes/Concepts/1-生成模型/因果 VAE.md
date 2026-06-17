---
type: concept
aliases: [因果 VAE, Causal VAE, 因果变分自编码器]
---

# 因果 VAE

## 定义
一种保持时序因果性的变分自编码器，在编解码视频序列时确保第 $t$ 帧的编码仅依赖第 $t$ 帧及之前的帧，常用于视频生成模型（如 Cosmos）的潜变量压缩。

## 核心要点
1. 因果卷积（causal convolution）保证时序单向依赖，避免未来帧"泄漏"到过去帧
2. 将高维 RGB 帧压缩为低维潜变量，降低扩散模型的去噪维度
3. 解码器可将潜变量重建为高分辨率图像/视频帧

## 数学形式

$$
z_t = \text{Enc}(o_{\leq t}), \quad \hat{o}_t = \text{Dec}(z_t)
$$

## 代表工作
- [[Cosmos Predict2]]: 使用因果 VAE 作为视频潜变量压缩模块
- [[NavWAM]]: 将机器人状态、动作等数值信号编码为图像格式后，经因果 VAE 压缩

## 相关概念
- [[Cosmos Predict2]]
- [[Diffusion Transformer]]
- [[Video Diffusion Model]]
