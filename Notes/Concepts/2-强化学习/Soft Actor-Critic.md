---
type: concept
aliases: [SAC, 软演员-评论家算法]
---

# Soft Actor-Critic

## 定义

基于最大熵框架的离策略深度强化学习算法，通过同时最大化预期收益和策略熵，实现稳定高效的连续动作空间控制。

## 数学形式

$$
\pi^* = \arg\max_{\pi} \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]
$$

其中 $\mathcal{H}$ 为策略熵，$\alpha$ 为温度系数（可自动调整）。

## 核心要点

1. 最大熵框架天然鼓励探索，避免过早收敛到次优策略
2. 离策略训练（replay buffer）使样本效率高于 PPO 等同策略方法
3. 双 Q 网络消除过估计偏差
4. 常用于连续控制任务的专家策略训练，为世界模型提供高质量演示数据

## 代表工作

- [[StableWorldModel]]: swm 使用 SAC 预训练专家策略，用于收集高质量训练轨迹

## 相关概念

- [[Model Predictive Control]]
- [[World Model]]
- [[GCBC]]
