---
type: concept
aliases: [元查询, meta-queries, 元查询机制]
---

# Meta-Query（元查询）

## 定义

Meta-Query 是一组可学习的查询向量，通过交叉注意力或因果注意力从输入序列中聚合关键信息，将高维上下文压缩为固定大小的紧凑表示，常用于多模态融合和条件生成。

## 数学形式

$$
\mathbf{h} = \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
$$

其中 $\mathbf{Q}$ 为可学习元查询，$\mathbf{K}, \mathbf{V}$ 来自输入上下文（图像、语言、状态等）。

## 核心要点

1. 固定数量的可学习向量（如 64 个），不随输入长度变化
2. 通过注意力机制聚合任意长度的上下文信息
3. 输出作为下游模块（世界模型、动作头）的条件信号
4. 类似 Perceiver / Q-Former 中的 latent query 设计

## 代表工作

- [[WLA]]: 使用 64 个元查询聚合观测、语言、历史，产生物理动态表示 $\mathbf{h}_t$

## 相关概念

- [[交叉注意力]]
- [[因果注意力]]
- [[自回归Transformer]]
