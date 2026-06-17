---
type: concept
aliases: [PlaNet, Planning with Latent Dynamics]
---

# PlaNet

## 定义
DeepMind 提出的基于模型的 RL 方法，通过学习潜空间动态模型（RSSM）在潜空间中进行 CEM 规划，是 DreamerV3 的前身。

## 数学形式
$$\text{RSSM}: h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}), \quad z_t \sim q_\phi(z_t|h_t, o_t)$$

## 核心要点
1. 循环状态空间模型（RSSM）建模潜动态
2. 在潜空间用 CEM 做轨迹优化
3. 仅从像素学习，无需真实奖励函数
4. DreamerV3 的直接前身

## 代表工作
- [[LoopWM]]: 以 PlaNet 为历史对比基线

## 相关概念
- [[DreamerV3]]
- [[RSSM]]
