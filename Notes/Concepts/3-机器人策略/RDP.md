---
type: concept
aliases: [Reactive Diffusion Policy, 反应式扩散策略]
---

# RDP（Reactive Diffusion Policy）

## 定义

一种将触觉和力信号作为额外观测输入与视觉特征被动融合的扩散策略基线，用于接触丰富操作任务。

## 核心要点

1. **被动融合**: 将多模态感知信号直接拼接输入策略，无显式时序预测
2. **扩散动作头**: 采用扩散过程生成动作序列

## 代表工作

- [[TacForeSight]]: 与 RDP 对比，在 Wire Insertion 等任务上 TacForeSight 大幅超越（RDP: 0% vs TacForeSight: 60%）

## 相关概念

- [[Diffusion Policy]]
- [[触觉世界模型]]
