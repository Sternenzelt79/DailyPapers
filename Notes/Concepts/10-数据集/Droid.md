---
type: concept
aliases: [Droid, DROID dataset]
---

# Droid

## 定义
一个大规模真实场景机器人操纵数据集（Distributed Robot Interaction Dataset），涵盖多种机器人、场景和任务，用于评估机器人泛化能力。

## 核心要点
1. 大规模真实场景数据，包含来自不同实验室（AUTOLab、CLVR、RAIL 等）的操作演示
2. 无 GT 深度，需依赖深度估计模型（如 Depth Anything V3）
3. 是评估视频世界模型在真实机器人操作中表现的重要 benchmark
4. 评估通常通过人类研究（human study）进行，关注任务成功、物体/机器人变形、指令遵循

## 代表工作
- [[GEM-4D]]: 在 400 个未见 Droid 样本上评估，成功率从 61% 提升至 81%（+20pp）

## 相关概念
- [[RLBench]]
- [[视频世界模型]]
- [[Depth Anything]]
