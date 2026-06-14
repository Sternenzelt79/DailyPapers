---
type: concept
aliases: [WISE framework, Weighted Iterative Self-Evolution]
---

# WISE

## 定义
用于多模态 Agent 强化学习训练的框架，通过迭代自进化和加权采样提升多 Agent 协作质量。

## 核心要点
1. 支持多 Agent pipeline 的联合 RL 训练
2. 用于 InterleaveThinker 等交错生成场景
3. 对不同质量的生成结果进行加权采样，改善训练信号

## 代表工作
- [[InterleaveThinker]]: 在交错文图生成任务中使用 WISE 进行 RL 训练

## 相关概念
- [[GRPO]]
- [[RISE]]
