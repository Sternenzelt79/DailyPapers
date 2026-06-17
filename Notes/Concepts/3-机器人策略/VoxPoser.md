---
type: concept
aliases: [VoxPoser, Voxel Composer]
---

# VoxPoser

## 定义
利用 LLM 合成 3D 体素值图（value map）作为操控约束，结合运动规划器生成机器人轨迹，无需任务特定训练数据。

## 核心要点
1. LLM 生成自然语言约束 → 转化为 3D 体素 affordance/avoidance map
2. 运动规划器在体素空间中搜索轨迹
3. 零样本泛化到新任务，无需额外数据收集

## 代表工作
- [[GeneralVLA-2]]: 使用 VoxPoser 提供语义约束

## 相关概念
- [[RoboPoint]]
- [[SAM]]
