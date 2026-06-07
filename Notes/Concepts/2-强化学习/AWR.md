---
type: concept
aliases: [Advantage Weighted Regression]
---

# AWR

## 定义
AWR（Advantage Weighted Regression）是一种 offline/off-policy 策略优化方法，通过对 behavior policy 采样的动作按 advantage 值加权回归来提升策略。

## 数学形式
$$\pi_{new} = \arg\max_\pi \mathbb{E}_{(s,a)\sim\pi_\beta}\left[\exp\left(\frac{A^\pi(s,a)}{\beta}\right)\log \pi(a|s)\right]$$

## 核心要点
1. 用 advantage 函数作为 importance weight，避免分布偏移
2. 相比 standard BC，AWR 可以过滤低质量数据
3. 不需要 Q-network 的 bootstrapping，稳定性好

## 代表工作
- Peng et al. (2019), AWR: Advantage-Weighted Regression for Offline RL

## 相关概念
- [[IDQL]]
- [[FQL]]
- [[Flow Matching]]
