---
type: concept
aliases: [Goal-Conditioned Behavioral Cloning, GCBC, 目标条件行为克隆]
---

# GCBC（Goal-Conditioned Behavioral Cloning）

## 定义
目标条件行为克隆，是模仿学习的一种变体：在给定目标状态（或图像）的条件下，直接从专家演示中学习策略 $\pi(a | s, g)$，无需奖励函数。

## 数学形式

$$
\mathcal{L}_{\text{GCBC}} = \mathbb{E}_{(s, a, g) \sim \mathcal{D}}\Big[\|\pi_\theta(s, g) - a\|^2\Big]
$$

## 核心要点
1. 最简单的离线 RL / 模仿学习方法，无需显式规划
2. 对专家数据质量敏感，分布外状态泛化差
3. 在 stable-worldmodel 评测中表现中等（Push-T: 75%，OGBench: 84%）
4. 常作为世界模型方法的弱基线对比

## 代表工作
- [[stable-worldmodel]]: 作为目标条件 RL 基线（2026）

## 相关概念
- [[GCIQL]]
- [[世界模型]]
- [[Reinforcement Learning]]
