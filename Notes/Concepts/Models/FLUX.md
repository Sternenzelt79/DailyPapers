---
type: concept
aliases: [FLUX, FLUX.1, Black Forest Labs]
---

# FLUX

## 定义
FLUX：Black Forest Labs（Stable Diffusion 原团队）推出的高质量图像生成模型系列，使用 [[MM-DiT]] 双流 Transformer 架构 + [[Flow Matching]]，在图像质量、文字渲染、指令跟随上大幅超越 SDXL。

## 数学形式
$$v_\theta(z_t, t, c) = (1-t) z_0 - t \epsilon, \quad z_t = (1-t)z_0 + t\epsilon$$

## 核心要点
1. 架构：[[MM-DiT]] 双流设计，text token 和 image token 各有独立 attention，再通过 cross-attention 交互
2. 训练范式：[[Flow Matching]] 替代 DDPM，采样更快、训练更稳
3. 版本：FLUX.1-dev（开放权重）、FLUX.1-schnell（蒸馏加速）、FLUX.1-pro（闭源最强）
4. FLUX.2 见 [[FLUX.2]]

## 代表工作
- Black Forest Labs 2024，FLUX.1 Technical Report

## 相关概念
- [[FLUX.2]]
- [[MM-DiT]]
- [[Flow Matching]]
- [[LDM]]
