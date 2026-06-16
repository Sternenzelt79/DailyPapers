---
type: concept
aliases: [IQL, Implicit Q-Learning]
---

# IQL（Implicit Q-Learning）

## 定义
IQL 是一种 offline RL 算法，通过 expectile regression 隐式估计最优 Q 值而无需显式对 OOD 动作求最大，避免了 offline RL 的分布外外推问题。

## 数学形式
$$\mathcal{L}_V(\psi) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\left[ L_2^\tau(Q_{\hat\theta}(s,a) - V_\psi(s)) \right]$$

其中 expectile loss 为：
$$L_2^\tau(u) = |\tau - \mathbb{1}[u<0]| \cdot u^2$$

## 核心要点
1. 拆分值函数估计为 V(s) 和 Q(s,a) 两步，V(s) 用 expectile regression 逼近 max_a Q
2. $\tau > 0.5$ 时 expectile 趋近最大值，隐式实现 offline policy improvement
3. 不需要 OOD 动作采样，避免 CQL 的保守性惩罚
4. 与 advantage weighted 方法兼容，常用于 offline-to-online RL 转换

## 代表工作
- IQL (Kostrikov et al., 2021): Offline Reinforcement Learning with Implicit Q-Learning

## 相关概念
- [[SAC]]
- [[Offline RL]]
