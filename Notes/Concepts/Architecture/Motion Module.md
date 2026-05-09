---
type: concept
category: Architecture
tags: [vla, motion-perception]
created: 2026-05-09
---

# Motion Module

视觉编码器中插入的运动感知模块，用于让 [[VLA]] 模型察觉场景中物体与机器人的相对运动。

典型实现是把 [[Space-Time Self-Similarity|STSS]] encoder 注入视觉 backbone 的中间层，通过计算 spatio-temporal token 与局部时空邻居的相关性产生运动感知特征，再通过 residual 加回主干。

## 代表工作

- [[RLDX-1]]: 在 27 层视觉 encoder 第 9 层后注入 STSS，用于 Conveyor Belt Catch 等高速运动任务（成功率 87.5%）
