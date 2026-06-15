---
type: concept
aliases: [Soft Actor-Critic, SAC算法]
---

# SAC（Soft Actor-Critic）

## 定义
基于最大熵强化学习框架的 off-policy actor-critic 算法，同时最大化累积奖励和策略熵。

## 数学形式
$$\pi^* = \arg\max_\pi \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}\left[r(s_t,a_t) + \alpha\mathcal{H}(\pi(\cdot|s_t))\right]$$

## 核心要点
1. 引入熵正则化项 $\alpha\mathcal{H}(\pi)$，鼓励探索，防止过早收敛
2. Off-policy 训练：利用经验回放缓冲区，样本效率高
3. 双 Q 网络（Clipped Double Q）缓解值函数高估问题
4. 自动调整温度系数 $\alpha$（Adaptive Temperature）

## 代表工作
- [[SAC]]: Haarnoja et al., 2018，最大熵RL框架奠基作
- [[EQRL]]: 在 VLA 弹性查询调度中使用 SAC 训练 difficulty estimator

## 相关概念
- [[Actor-Critic]]
- [[最大熵强化学习]]
- [[PPO]]
