---
type: concept
aliases: [AdaPool, Adaptive Pooling]
---

# AdaPool

## 定义
自适应池化（Adaptive Pooling）操作，根据输入内容动态调整池化权重，将可变长度的 token 序列压缩为固定大小的表示，常用于将潜空间 token 聚合为标量预测（如奖励值）。

## 数学形式
$$
\mathbf{h} = \sum_i \alpha_i \mathbf{z}_i, \quad \alpha_i = \text{softmax}(f(\mathbf{z}_i))
$$

其中 $f$ 为可学习的权重预测网络，$\alpha_i$ 为各 token 的注意力权重。

## 核心要点
1. 相比全局平均池化，AdaPool 可以学习关注最相关的 token
2. 适合从高维 token 序列中提取任务相关信号（如语言-条件奖励）
3. 输出维度固定，便于接后续 MLP 头

## 代表工作
- [[WEAVER]]: 奖励头中使用 AdaPool 将潜向量 token 聚合后预测每步奖励

## 相关概念
- [[Reward Model]]
- [[Transformer]]
