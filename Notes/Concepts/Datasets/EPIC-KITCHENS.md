---
type: concept
aliases: [EPIC-KITCHENS-100, EPIC Kitchens]
---

# EPIC-KITCHENS

## 定义

EPIC-KITCHENS 是一个大规模自我中心（第一人称）厨房活动视频数据集，通过可穿戴摄像机记录参与者在厨房中的日常操作活动。

## 核心要点

1. **规模**：EPIC-KITCHENS-100 包含约 100 小时视频，涵盖 700 多个参与者-场景组合
2. **视角**：纯第一人称（自我中心）视角，使用头戴式 GoPro 摄像机
3. **动作标注**：精细动作标签（动词+名词，如"cut tomato"），包含时间戳、边界框等
4. **任务**：动作识别、动作检测、目标检测、视频检索等
5. **局限**：仅限厨房场景，规模（100h）远小于 HumanNet（1,000,000h）

## 代表工作

- [[HumanNet]]: 将 EPIC-KITCHENS 作为对比基线，规模超越其约 10,000 倍

## 相关概念

- [[Ego4D]]
- [[EgoScale]]
- [[Vision-Language-Action Model]]
