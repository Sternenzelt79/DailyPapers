---
type: concept
aliases: [CEM, Cross-Entropy Method]
---

# Cross-Entropy Method (CEM)

## 定义
基于采样的随机优化器：从一个参数化分布（通常高斯）采样候选解，按代价排序，用 top-$k$ "elites" 重新拟合分布，迭代直到收敛。Rubinstein, 1999。

## 数学形式
$$
\mu^{(k+1)}, \Sigma^{(k+1)} = \mathrm{fit}\!\big(\{a^{(i)}\}_{i \in \text{elite}}\big),\quad a^{(i)} \sim \mathcal{N}(\mu^{(k)}, \Sigma^{(k)})
$$

## 核心要点
1. 在机器人 [[Model Predictive Control|MPC]] 中常用作 trajectory optimizer，无需可微动力学。
2. 主要超参：样本数、elite 比例、迭代次数、初始协方差。
3. [[LeWM]] 用 300 samples × 30 iters，elite=30。

## 代表工作
- iCEM (Pinneri 2020)
- TD-MPC, [[DINO-WM]], [[LeWM]] 的规划器

## 相关概念
- [[Model Predictive Control]]
- [[MPPI]]
