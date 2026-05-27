---
type: concept
aliases: [CogACT]
---

# CogACT

## 定义
一种 VLA 模型，以 CogVLM 为骨干，集成 action chunking transformer 做连续动作预测，广泛被 VLA 加速/压缩工作用作 baseline。

## 核心要点
1. **CogVLM 骨干**：利用 CogVLM 的视觉-语言理解能力做 embodied 任务
2. **Action Head**：采用 [[ACT]] 风格的 action chunking，预测多步动作序列
3. **常用 Baseline**：[[SpecPrune-VLA]]、[[DySta]] 等 VLA 效率工作均与 CogACT 对比

## 代表工作
- Liu et al.《CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation》

## 相关概念
- [[VLA]]
- [[ACT]]
- [[Action Chunking]]
- [[OpenVLA]]
