---
type: concept
aliases: [UniZero算法]
---

# UniZero

## 定义
基于 MuZero 框架的统一世界模型 RL 算法，将表征学习与规划集成到单一 transformer 架构中，消除 MuZero 中独立的表征/动态网络分离。

## 核心要点
1. 用单一 Transformer 同时承担表征学习和动态预测（替代 MuZero 的双网络设计）
2. 在连续控制和离散决策任务上统一表现
3. Context-aware 的历史建模使长程规划更稳定
4. 在 Atari 100K 和 DMControl 上达到 SOTA

## 代表工作
- [[UniZero]]: Ye et al., 2024
- [[COMET]]: 对比基线，在 ManiSkill 上与 UniZero 对比

## 相关概念
- [[MuZero]]
- [[MBRL]]
- [[MCTS]]
