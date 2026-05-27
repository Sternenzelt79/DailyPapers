---
type: concept
aliases: [Sparse VLM, Text-to-Vision Pruning]
---

# SparseVLM

## 定义

一种 VLM 视觉 Token 剪枝方法，基于文本-视觉（Text-to-Vision）跨注意力分数衡量视觉 Token 的语义重要性并进行剪枝。

## 核心要点

1. **Text-to-Vision 注意力**: 利用文本 Token 对视觉 Token 的注意力权重衡量视觉 Token 与文本语义的相关性
2. **语义中心**: 与 [[FastV]] 类似，以语义相关性为唯一标准，忽视动作相关性
3. **需要校准**: 相比 Training-Free 方法，SparseVLM 可能需要额外校准步骤

## 局限性

直接用于 VLA 推理时，因忽视 Action Decode 阶段的视觉局部需求，高剪枝率下性能退化显著（LIBERO 87.5% 剪枝率下相对精度仅 63.2%）。

## 代表工作

- [[VLA-Pruner]]: 将 SparseVLM 作为基线对比，证明仅靠语义注意力不足以支撑 VLA 操控任务

## 相关概念

- [[视觉 Token 剪枝]]
- [[FastV]]
- [[交叉注意力]]
