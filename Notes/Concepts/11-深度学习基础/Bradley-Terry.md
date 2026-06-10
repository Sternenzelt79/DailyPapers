---
type: concept
aliases: [Bradley-Terry Model, BT模型, 配对比较排名]
---

# Bradley-Terry 模型

## 定义

一种统计排名模型，通过两两配对比较（pairwise comparison）的结果估计每个候选项的潜在能力分数，并从中推断全局排名。

## 数学形式

给定选手 $i$ 与 $j$ 的对战，$i$ 获胜的概率为：

$$
P(i \text{ beats } j) = \frac{\exp(\beta_i)}{\exp(\beta_i) + \exp(\beta_j)}
$$

其中 $\beta_i, \beta_j$ 为各候选项的潜在能力分数，通过最大似然估计拟合。

## 核心要点

1. 只需要两两比较的胜负结果，无需绝对评分
2. 输出全局排名分数，可处理传递性不一致（A>B, B>C, C>A）
3. 常用于 LLM 评估、AlphaGo/ELO 排名等场景

## 代表工作

- [[OSCAR]]: 在 RoboArena 策略评估中，使用 Bradley-Terry 从 65 个 session × 7 个策略的配对结果估计策略排名，与真实机器人评估结果计算 MMRV 等相关性指标

## 相关概念

- [[Elo Rating]]
