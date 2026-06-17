---
type: concept
aliases: [Flow-based Layer-wise Activation Pruning]
---

# FLAP

## 定义
基于激活值分析的层级剪枝方法，用于压缩大型语言/多模态模型，在 VLA 压缩场景中被广泛用作基线。

## 核心要点
1. 分析各层激活值的重要性进行选择性剪枝
2. 结构化剪枝，保持硬件友好性
3. 在 VLA 场景下与 EfficientVLA、LightVLA 配合使用

## 代表工作
- [[RLRC]]: 以 FLAP 作为压缩方法之一

## 相关概念
- [[EfficientVLA]]
- [[LightVLA]]
