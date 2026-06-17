---
type: concept
aliases: [AWR, Advantage-Weighted Regression, 优势加权回归]
---

# 优势加权回归（AWR）

## 定义

优势加权回归（AWR）是一种离线/在线 RL 方法，通过对行为克隆损失中的每个 transition 按优势函数 $A(s,a)$ 加权，实现对"好动作"的强化学习，无需显式策略梯度。

## 数学形式

$$
\mathcal{L}_{\text{AWR}} = -\mathbb{E}_{(s,a) \sim \mathcal{D}}\left[\exp\left(\frac{A(s,a)}{\beta}\right) \log \pi_\theta(a|s)\right]
$$

其中 $\beta$ 控制温度（权重的尖锐程度）。

## 核心要点

1. **隐式策略改进**: 通过加权 BC 近似策略梯度更新，避免 on-policy 采样不稳定
2. **优势函数**: $A(s,a) = Q(s,a) - V(s)$ 衡量动作相对于基线的优劣
3. **指数加权**: 好动作（$A > 0$）权重指数放大，差动作（$A < 0$）权重压缩
4. **HABC 的区别**: HABC 用 $g_t = 1 + \tanh(\cdot)$ 替代指数加权，将权重限制在 $(0,2)$ 避免权重爆炸

## 代表工作

- [[HABC]]: 层次优势加权行为克隆，AWR 的双目标扩展版本
- [[AWAC]]: AWR 的在线数据采集改进版

## 相关概念

- [[强化学习]]
- [[加权行为克隆]]
- [[Recap]]
- [[双头Critic]]
