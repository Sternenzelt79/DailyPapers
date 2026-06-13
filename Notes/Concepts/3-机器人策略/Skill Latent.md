---
type: concept
aliases: [技能潜变量, 技能潜向量, Skill Latent, Skill Token]
---

# Skill Latent

## 定义

将一段连续的低级动作序列（运动原语）压缩成单一高维向量的表示，捕获语义上连贯的"技能"（如"抓取物体"、"移动到目标位置"），用于高层规划。

## 数学形式

$$
z^h_j = \sum_{i \in \mathcal{I}_j} \alpha_{j,i} h_i, \quad \alpha_{j,i} = \frac{\exp(w^\top h_i)}{\sum_{k \in \mathcal{I}_j} \exp(w^\top h_k)}
$$

其中 $\mathcal{I}_j$ 为第 $j$ 个技能片段的时间步集合，$h_i$ 为低级 token，$w$ 为可学习池化权重。

## 核心要点

1. **可变长度**: 不同技能的执行时长不同，技能潜变量通过注意力池化压缩可变长度片段
2. **层级结构**: 低级潜动作 → 技能潜变量 → 任务完成，形成从细粒度到粗粒度的层级
3. **规划接口**: 技能潜变量作为规划器（高层）与执行器（低层）之间的接口
4. **可复用性**: 学习到的技能潜变量可在类似任务中复用

## 代表工作

- [[HiMem-WAM]]: 通过多层动态分块发现技能边界，学习技能潜变量作为高层规划单元

## 相关概念

- [[Hierarchical Latent Action]]
- [[Latent Action]]
- [[Dynamic Chunking]]
- [[Temporal Abstraction]]
