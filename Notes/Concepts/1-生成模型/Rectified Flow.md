---
type: concept
aliases: [整流流, RF, Flow Matching, 整流流匹配]
---

# Rectified Flow

## 定义

一种生成模型训练框架，通过在噪声 $\epsilon$ 和真实数据 $z_0$ 之间学习线性插值路径上的速度场，使得 ODE 轨迹尽量"笔直"，从而提升采样效率。

## 数学形式

**线性插值路径**:
$$
z_t = (1-t)z_0 + t\epsilon, \quad t \in [0, 1]
$$

**训练目标（速度场预测）**:
$$
\mathcal{L}_{\text{RF}} = \mathbb{E}_{t,z_0,\epsilon}\left[\| v_\theta(z_t, t, c) - (\epsilon - z_0) \|_2^2\right]
$$

**推理**（ODE 积分）:
$$
z_{t+\Delta t} = z_t + v_\theta(z_t, t, c) \cdot \Delta t
$$

## 核心要点

1. 与 DDPM 不同，目标速度场是确定的线性方向 $(\epsilon - z_0)$，无需估计 score function
2. 直线轨迹意味着更少的 NFE（函数评估次数）即可完成采样
3. [[Diffusion Transformer]]（DiT）常以 Rectified Flow 为训练目标

## 代表工作

- [[OSCAR]]: 以 Rectified Flow 为目标微调 Cosmos-Predict2.5-2B，用于动作条件视频生成
- SD3、Flux、Wan、Cosmos 等现代视频/图像生成模型的标准训练目标

## 相关概念

- [[Diffusion Transformer]]
- [[VAE]]
- [[CFM]]
