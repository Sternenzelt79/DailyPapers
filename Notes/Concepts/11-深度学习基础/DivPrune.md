---
type: concept
aliases: [Diversity-based Pruning, 多样性剪枝]
---

# DivPrune

## 定义

一种基于特征多样性的 VLM 视觉 Token 剪枝方法，通过最大化所选 Token 集合的特征多样性来减少冗余并保留代表性视觉信息。

## 核心要点

1. **多样性驱动**: 以 Token 特征之间的距离为选择标准，倾向于保留彼此差异大的 Token
2. **缺乏动作感知**: 仅基于特征多样性，未建模 VLA 推理中动作解码阶段对局部视觉信息的特定需求
3. **与语义剪枝互补**: 相比 [[FastV]] 的注意力分数，DivPrune 更关注信息不冗余性

## 局限性

直接用于 VLA 推理时，由于忽视动作相关 Token 的重要性，在高剪枝率下性能显著退化（LIBERO 87.5% 剪枝率下相对精度仅 58%）。

## 代表工作

- [[VLA-Pruner]]: 将 DivPrune 作为基线，通过 Combine-then-Filter 策略在多样性基础上引入语义-动作双重重要性估计

## 相关概念

- [[视觉 Token 剪枝]]
- [[FastV]]
- [[Combine-then-Filter]]
