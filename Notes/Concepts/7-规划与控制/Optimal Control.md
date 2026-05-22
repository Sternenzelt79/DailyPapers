---
type: concept
aliases: [最优控制, 最优控制问题, Optimal Control Problem]
---

# Optimal Control

## 定义

在给定动力学约束下，找到使累积代价函数最小化的控制策略，是控制理论的核心问题之一。

## 数学形式

$$
\pi^* = \arg\min_{\pi} \sum_{t=0}^{T} c(s_t, a_t)
$$

$$
\text{s.t.} \quad s_{t+1} = f(s_t, a_t), \quad a_t = \pi(s_t)
$$

其中 $f$ 为系统动力学（或学习到的世界模型 $P$），$c$ 为阶段代价函数，$T$ 为规划时域。

## 核心要点

1. 分为离线（学习全局策略 $\pi$）和在线（实时优化，即 MPC）两类求解范式
2. 有限时域版本对应 MPC，在测试时直接优化动作序列
3. 世界模型将 $f$ 替换为学习到的预测器 $P$，使从数据中学习规划成为可能
4. 求解方法：采样方法（CEM、MPPI）、梯度方法（PG、GRASP）

## 代表工作

- [[StableWorldModel]]: 将最优控制形式化为 swm 评估框架的理论基础
- [[Model Predictive Control]]: 最优控制的在线求解范式
- [[LeWM]]: 端到端世界模型通过近似求解最优控制进行规划

## 相关概念

- [[Model Predictive Control]]
- [[Cross-Entropy Method]]
- [[MPPI]]
- [[World Model]]
