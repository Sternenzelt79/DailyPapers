---
type: concept
aliases: [Temporal Difference Model Predictive Control, TD-MPC2]
---

# TD-MPC

## 定义
Temporal Difference Model Predictive Control，结合 TD 学习的隐式世界模型与 MPC 在线规划，无需像素重建，直接在学习的潜在空间中做滚动优化。

## 数学形式
$$J(\pi) = \mathbb{E}\left[\sum_{t=0}^{H} \gamma^t r(z_t, a_t) + \gamma^{H+1} V(z_{H+1})\right]$$

其中 $z_t$ 为潜在状态，$V$ 为值函数，$H$ 为规划步长。

## 核心要点
1. 无重建损失：encoder 和动力学模型端到端通过 TD 目标训练
2. 比 Dreamer 系列更 sample efficient，避免了像素解码器开销
3. 潜在空间 MPC 结合值函数，比纯 MPC 更长视野

## 代表工作
- Hansen et al. (2022/2023): TD-MPC / TD-MPC2
- [[SMWM]]: 对比基线

## 相关概念
- [[JEPA]]
- [[RSSM]]
- [[SMWM]]
