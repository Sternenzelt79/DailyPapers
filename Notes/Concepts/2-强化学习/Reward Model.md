---
type: concept
aliases: [Reward Model, 奖励模型, Reward Head]
---

# Reward Model

## 定义
在强化学习和世界模型中，奖励模型是一个学习预测每个状态（或状态-动作对）奖励信号的网络，可替代真实环境的奖励函数，用于策略评估和优化。

## 数学形式
$$
\hat{r}_t = R(z_t, \ell), \quad \mathcal{L} = \|R(z_t, \ell) - r_t\|^2_2
$$

其中 $z_t$ 为潜状态，$\ell$ 为语言条件，$r_t$ 为真实奖励标签（如 Robometer 评分）。

## 核心要点
1. 解耦奖励信号与策略，允许在潜空间中高效评估
2. 语言条件奖励模型支持多任务泛化
3. 可与 Critic 网络结合实现长程价值估计

## 代表工作
- [[WEAVER]]: 奖励头通过 AdaPool 聚合多视角潜 token，在世界模型潜空间中直接预测 Robometer 奖励

## 相关概念
- [[Critic Network]]
- [[Monte Carlo Advantage]]
- [[TD Lambda]]
- [[AdaPool]]
