---
type: concept
aliases: [Diffusion Forcing, 扩散强迫]
---

# Diffusion Forcing

## 定义
将扩散模型与自回归序列模型统一的训练框架：每个 token 在训练时被赋予独立随机的噪声水平，推理时既可以做完整去噪（生成），也可以做有条件预测（规划/控制）。

## 数学形式
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}, \boldsymbol{\epsilon}, \mathbf{n}} \left[ \| \boldsymbol{\epsilon} - \epsilon_\theta(\mathbf{x}_\mathbf{n} + \boldsymbol{\epsilon}, \mathbf{n}, t) \|^2 \right]$$

其中 $\mathbf{n} \in \mathbb{R}^T$ 是每个 token 的独立噪声水平向量，$\mathbf{x}_\mathbf{n}$ 是按噪声水平加噪后的序列。

## 核心要点
1. **统一 AR + 扩散**：单一框架同时支持自回归生成和多步去噪，无需分开训练两个模型
2. **灵活噪声调度**：推理时可以对过去帧用低噪声（条件）、未来帧用高噪声（生成），实现滚动预测
3. **规划应用**：可直接用于 MPC/CEM 规划，对未来序列做条件生成即可得到候选轨迹
4. **世界模型基础**：[[NanoWM]] 等以 Diffusion Forcing 作为统一世界模型框架

## 代表工作
- Chen et al. 2024《Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion》
- [[NanoWM]]：以 Diffusion Forcing 为核心的极简世界模型实现

## 相关概念
- [[Diffusion Model]]
- [[Autoregressive Policy]]
- [[DDPM]]
- [[World Model]]
