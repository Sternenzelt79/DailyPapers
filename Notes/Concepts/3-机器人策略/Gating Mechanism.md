---
type: concept
aliases: [门控机制, 自适应门控, Adaptive Gating]
---

# Gating Mechanism

## 定义

一种动态权重分配机制，通过可学习或条件化的门控值控制不同信息流的融合比例，广泛用于多模态融合、时序建模和注意力调制。

## 数学形式

$$
\boldsymbol{\alpha} = \sigma(\text{MLP}(\mathbf{h}_{cond}))
$$

$$
\mathbf{h}_{out} = (1 - \boldsymbol{\alpha}) \odot \mathbf{h}_A + \boldsymbol{\alpha} \odot \mathbf{h}_B
$$

## 核心要点

1. **条件化门控**: 门控权重由某一模态特征条件化生成，实现自适应融合
2. **元素级控制**: $\boldsymbol{\alpha}$ 可以是标量（全局门控）或向量/矩阵（元素级精细控制）
3. **可解释性**: 门控值可视化有助于理解模型对不同模态的依赖程度

## 代表工作

- [[TacForeSight]]: 用触觉特征条件化门控权重，接触强度大时触觉主导，无接触时视觉主导

## 相关概念

- [[Cross-Attention]]
- [[Transformer]]
