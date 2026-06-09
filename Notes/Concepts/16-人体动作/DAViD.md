---
type: concept
aliases: [DAViD, 视频驱动HOI生成]
---

# DAViD

## 定义

一种基于视频先验的 4D 人-物交互（HOI）生成方法，通过从视频中重建人体与物体运动来生成操作示范。

## 核心要点

1. 使用视频作为交互行为先验，无需显式 3D 场景建模
2. 缺乏完整 3D 场景约束，重建尺度存在歧义
3. 在 GRAIL 对比中：追踪成功率 24.0%（vs GRAIL 88.9%）；Contact Distance 0.246（vs GRAIL 0.008）

## 代表工作

- [[GRAIL]]: DAViD 作为 HOI 生成对比基线，GRAIL 在所有指标上显著超越

## 相关概念

- [[HOI]]
- [[视频基础模型]]
- [[HOIDiff]]
- [[CHOIS]]
