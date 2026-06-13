---
type: concept
aliases: [Monte Carlo Advantage, 蒙特卡洛优势, MC Advantage]
---

# Monte Carlo Advantage

## 定义
通过完整或截断的蒙特卡洛轨迹回报减去基线价值估计，得到动作相对于平均水平的优势信号，用于策略梯度或动作选择。

## 数学形式
$$
\hat{A}^b_t = \sum_{\ell=1}^{H} \gamma^{\ell-1} r^b_{t+\ell} + \gamma^H V(s_{t+H}) - V(s_t)
$$

在世界模型中，奖励由模型预测替代：
$$
\hat{A}^b_t = \sum_{\ell=1}^{H} \gamma^{\ell-1} R(\hat{z}^b_{t+\ell}, \ell) + \gamma^H V(\hat{z}^b_{t+H}, \ell) - V(z_t, \ell)
$$

## 核心要点
1. **无偏但高方差**: 蒙特卡洛回报无偏，但完整轨迹累计噪声方差大
2. **价值基线**: 减去 $V(s_t)$ 可显著降低方差
3. **多轨迹对比**: 采样多条轨迹后选优势最高者，是 Best-of-N 规划的核心

## 代表工作
- [[WEAVER]]: 使用 MC 优势在 B=4 条并行想象轨迹中筛选最优动作块

## 相关概念
- [[Critic Network]]
- [[TD Lambda]]
- [[Reward Model]]
