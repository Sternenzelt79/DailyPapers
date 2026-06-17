---
type: concept
aliases: [策略梯度, REINFORCE, Policy Gradient Method]
---

# Policy Gradient

## 定义

策略梯度方法直接对策略参数求梯度，通过最大化期望累积奖励来优化策略，是一类基于梯度的强化学习算法基础。

## 数学形式

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t\right]
$$

其中优势函数 $A_t = R_t - b(s_t)$，$b$ 为基线函数（通常用值函数 $V(s_t)$）。

## 核心要点

1. **直接优化策略**：不需要显式构建值函数，直接对策略参数梯度上升
2. **优势函数加权**：用优势函数（而非原始回报）减小方差、加速收敛
3. **对数技巧**：$\nabla \log \pi = \nabla\pi / \pi$，将期望梯度转化为样本均值估计
4. **高方差问题**：蒙特卡洛估计方差大，实践中需结合基线函数、Actor-Critic 等技巧

## 代表工作

- [[WAM-RL]]: 将策略梯度用于 Flow-SDE 参数化的动作模型 RL 训练
- [[PPO]]: 加入 clip 约束的改进版策略梯度
- [[GRPO]]: 去掉 Critic 的组相对策略优化

## 相关概念

- [[Reinforcement Learning]]
- [[PPO]]
- [[Actor-Critic]]
- [[GRPO]]
- [[Flow Matching]]
