---
type: concept
aliases: [Recap, TD Residual Filtering]
---

# Recap

## 定义

Recap 是一种基于 TD（时序差分）残差的**硬阈值过滤**在线 RL 方法，通过设定固定阈值筛选"有价值"的 transition 用于策略更新，丢弃 TD 残差低于阈值的样本。

## 数学形式

对每个 transition $(s_t, a_t, s_{t+1})$，计算 TD 残差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

仅保留 $\delta_t > \tau$（固定阈值）的 transition 参与 actor 更新。

## 核心要点

1. **硬阈值过滤**: 以固定阈值 $\tau$ 为判断标准，低于阈值的 transition 直接丢弃（pass rate 约 19-27%）
2. **局限**: 丢弃了大量有效的负向学习信号，可能损失策略改进信息
3. **与 HABC 对比**: HABC 用软加权（$g_t \in (0,2)$）替代硬过滤，保留更多梯度信息

## 代表工作

- [[HABC]]: 将 Recap 作为基线对比方法（Imit-Recap），证明软加权优于硬过滤

## 相关概念

- [[优势加权回归（AWR）]]
- [[加权行为克隆]]
- [[双头Critic]]
