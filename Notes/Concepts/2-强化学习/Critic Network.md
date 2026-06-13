---
type: concept
aliases: [Critic Network, 价值网络, Value Network]
---

# Critic Network

## 定义
在 Actor-Critic 框架中，Critic 网络（价值网络）负责估计状态价值函数 $V(s)$ 或状态-动作价值函数 $Q(s,a)$，为策略梯度提供低方差基线或引导优势估计。

## 数学形式
$$
V(s_t) \approx \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k}\right]
$$

结合 TD-λ 训练时：
$$
\mathcal{L}^{\text{critic}} = \left\| V(z_t, \ell) - \mathbf{v}^\lambda_t \right\|^2_2
$$

## 核心要点
1. 提供价值基线，降低策略梯度估计的方差
2. 可用于估计超出想象地平线的价值（bootstrapping）
3. 与奖励模型配合使用，实现长程价值分解

## 代表工作
- [[WEAVER]]: Critic 网络基于潜空间状态和语言指令预测 λ-return，支持测试时规划

## 相关概念
- [[TD Lambda]]
- [[Monte Carlo Advantage]]
- [[Reward Model]]
