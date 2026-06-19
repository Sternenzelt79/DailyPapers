---
type: concept
aliases: [Model-Free Reinforcement Learning, 无模型强化学习]
---

# MFRL

## 定义
Model-Free Reinforcement Learning，不显式学习环境动力学模型，直接从与环境的交互中学习值函数或策略的强化学习范式。

## 数学形式
策略梯度目标：
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t r_t\right]$$

## 核心要点
1. 无需准确的世界模型，避免了模型误差累积
2. 通常 sample efficiency 低于 model-based 方法
3. 代表算法：PPO、SAC、TD3、A3C

## 代表工作
- [[SWAP]]: MFRL for legged locomotion 基线对比

## 相关概念
- [[WAM]]
- [[RSSM]]
- [[SAC]]
