---
type: concept
aliases: [Twin RL, TwinRL, Twin Reinforcement Learning]
---

# TwinRL

## 定义
双网络强化学习方法，通过维护两个相互独立的策略网络相互监督/校正，用于提高机器人 RL 训练的稳定性。

## 核心要点
1. 双策略网络架构减少 value overestimation
2. 在机器人操控 RL 训练中提升稳定性
3. WAM-RL 中作为对比基线

## 代表工作
- [[WAM-RL]]: 与 TwinRL 对比世界模型增强的 RL

## 相关概念
- [[ConRFT]]
- [[PPO]]
