---
type: concept
aliases: [MPC, Model Predictive Control, Receding Horizon Control]
---

# Model Predictive Control (MPC)

## 定义
在每一控制时刻，用一个动力学模型在线求解一个有限时域最优控制问题，执行第一步（或前 $K$ 步）动作后再重规划。

## 数学形式
$$
a^*_{1:H} = \arg\min_{a_{1:H}} \sum_{t=1}^{H} c(\hat z_t, a_t) + c_T(\hat z_H)
$$
执行 $a^*_{1:K}$，然后从新的观测重启。

## 核心要点
1. 滚动时域 + 闭环；动力学可学（[[World Model]]）也可解析（机器人 dyn）。
2. 求解器：CEM / [[MPPI]] / iLQR / DDP / 直接微分。
3. [[LeWM]] 用 latent-space MPC + CEM。

## 代表工作
- [[LeWM]]：latent CEM-MPC
- TD-MPC2
- 经典控制：QP/iLQR-based MPC

## 相关概念
- [[Cross-Entropy Method]]
- [[World Model]]
