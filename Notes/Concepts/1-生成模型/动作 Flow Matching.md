---
type: concept
aliases: [Action Flow Matching, 动作流匹配]
---

# 动作 Flow Matching

## 定义

动作 Flow Matching 是将 [[Conditional Flow Matching]] 框架应用于机器人动作序列生成的技术，以 trace 引导特征为条件，训练动作去噪网络生成控制序列。

## 数学形式

$$
\mathcal{L}_{\text{action}} = \mathbb{E}_{\tau, \mathbf{a}, \boldsymbol{\epsilon}_a} \left\| v_{\phi}\!\left(\mathbf{a}^{\tau}, \tau, \mathbf{z}_{\text{guided}}, \mathbf{c}\right) - (\mathbf{a} - \boldsymbol{\epsilon}_a) \right\|_{2}^{2}
$$

## 核心要点

1. **条件生成**: 以 trace 引导特征 $\mathbf{z}_{\text{guided}}$ 和任务条件 $\mathbf{c}$ 为输入
2. **与轨迹流匹配分离**: 动作专家单独训练，μ₀ 主干冻结
3. **实体特定**: 不同机器人平台各训练独立的动作专家头

## 代表工作

- [[mu0]]: 动作专家用动作 Flow Matching 训练，实现从冻结 μ₀ trace 特征到机器人控制的适配

## 相关概念

- [[Conditional Flow Matching]]
- [[Flow Matching]]
- [[Gated Cross-Attention]]
