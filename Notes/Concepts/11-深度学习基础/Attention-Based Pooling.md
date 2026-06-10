---
type: concept
aliases: [注意力池化, 注意力聚合, Attention-Based Pooling, Attentive Pooling]
---

# Attention-Based Pooling

## 定义

使用可学习的注意力权重对序列中的多个 token 进行加权求和，生成单一聚合表示，相比均值池化/最大池化能自适应地聚焦重要元素。

## 数学形式

$$
\alpha_i = \frac{\exp(w^\top h_i)}{\sum_{k} \exp(w^\top h_k)}, \quad z = \sum_i \alpha_i h_i
$$

其中 $w$ 为可学习权重向量，$h_i$ 为第 $i$ 个 token 的隐状态。

在分段池化（用于 [[Dynamic Chunking]]）中，只对片段 $\mathcal{I}_j$ 内的 token 归一化：

$$
z^{(s+1)}_j = \sum_{i \in \mathcal{I}_j} \alpha_{j,i} h^{(s)}_i
$$

## 核心要点

1. **可变长度输入**: 可以处理任意长度的序列片段，输出固定维度
2. **可学习聚焦**: 学习对序列中最具代表性的 token 赋予更高权重
3. **端到端可训练**: 注意力权重通过反向传播学习
4. **分段应用**: 结合动态分块，可对每个技能片段独立池化

## 代表工作

- [[HiMem-WAM]]: 对每个技能边界内的低级潜动作 token 进行注意力池化，生成技能潜变量

## 相关概念

- [[Dynamic Chunking]]
- [[Skill Latent]]
- [[Hierarchical Latent Action]]
