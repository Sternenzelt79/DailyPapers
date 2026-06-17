---
type: concept
aliases: [DreamerV2, Dreamer V2]
---

# DreamerV2

## 定义
DreamerV2 是基于离散 latent 表示的 world model + actor-critic 框架，在 Atari 上首次超过人类基准，无需像素级 reward，通过模型内部想象完成策略学习。

## 数学形式
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}} + \mathcal{L}_{\text{reward}}$$

World model 由 RSSM（Recurrent State Space Model）建模：
$$h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}), \quad z_t \sim q_\phi(z_t | h_t, x_t)$$

## 核心要点
1. 用 VQ-VAE 风格的 straight-through 梯度离散化 latent，解决 KL 坍缩
2. RSSM 分离确定性（RNN cell）和随机性（stochastic state）两条路径
3. Actor-critic 完全在 world model 想象的 latent 轨迹中训练，不接触真实环境
4. 比 DreamerV1 稳定，比 model-free 方法样本效率高 10-100x

## 代表工作
- [[DreamerV2]] (Hafner et al., 2021): Mastering Atari with Discrete World Models

## 相关概念
- [[DreamerV3]]
- [[RSSM]]
- [[World Model]]
