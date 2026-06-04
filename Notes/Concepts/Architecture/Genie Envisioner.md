---
type: concept
aliases: [Genie Envisioner]
category: Architecture
tags: [world-model, video-generation, gripper-rendering, robot-manipulation]
created: 2026-06-04
---

# Genie Envisioner

基于夹爪渲染（Gripper Rendering）条件的机器人操作视频世界模型，参数量 2B，在 AgiBot World 数据集上训练。是 OSCAR 最接近的基线之一。

## 核心特点

- **条件类型**: 夹爪图像渲染（Gripper Rendering）
- **参数量**: 2B
- **训练数据**: AgiBot World
- **性能**: PSNR 23.29，SSIM 0.838，FVD 15.37

## 对比

与 [[OSCAR]] 相比：OSCAR 在所有指标上超越（PSNR +0.95 dB，FVD 7.08 vs 15.37），且支持跨形态泛化（夹爪渲染与特定机器人形态强绑定）。

## 相关概念

- [[Action-Conditioned Video World Model]]: 所属任务类别
- [[OSCAR]]: 同参数量下性能更优的后续工作
