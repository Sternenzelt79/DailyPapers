---
type: concept
aliases: [OGBench, OGBench-Cube]
---

# OGBench

## 定义
Offline goal-conditioned RL benchmark，包含 navigation / manipulation / locomotion 子任务。OGBench-Cube 是其中的 3D 魔方操作子集。

## 核心要点
1. 提供 offline、reward-free 数据，强调 goal-conditioned 设定。
2. OGBench-Cube 涉及 3D 旋转，用来测视觉-动作世界模型的 3D 推理能力。
3. [[LeWM]] / [[DINO-WM]] / [[PLDM]] 在此对比。

## 代表工作
- OGBench (Park 2024)
- [[LeWM]]：在 OGBench-Cube 上做 3D 规划评估

## 相关概念
- [[World Model]]
- Goal-Conditioned RL
