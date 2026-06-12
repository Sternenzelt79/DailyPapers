---
type: concept
aliases: [TD Lambda, TD-λ, Lambda Return, λ-return]
---

# TD Lambda（λ-Return）

## 定义
时序差分学习（Temporal Difference Learning）与蒙特卡洛估计的加权混合，通过参数 λ 控制在 n 步回报中偏向 TD（低方差高偏差）还是 MC（无偏高方差）。

## 数学形式
$$
\mathbf{v}^\lambda_t = r_t + \gamma\left[(1-\lambda)V(s_{t+1}) + \lambda \mathbf{v}^\lambda_{t+1}\right]
$$

展开后等价于无穷级数：
$$
\mathbf{v}^\lambda_t = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}
$$

其中 $G_t^{(n)}$ 为 n 步回报。

## 核心要点
1. **λ=0**: 纯 TD(0)，一步自举，低方差高偏差
2. **λ=1**: 等价于蒙特卡洛，无偏差但高方差
3. **λ=0.95**: 实践中常用的平衡点
4. **Critic 训练**: 用 λ-return 作为目标训练价值网络，比纯 TD 更稳定

## 代表工作
- [[WEAVER]]: Critic 网络使用 γ=0.995、λ=0.95 的 TD-λ 目标，在潜空间中估计长程价值

## 相关概念
- [[Critic Network]]
- [[Monte Carlo Advantage]]
- [[Reward Model]]
