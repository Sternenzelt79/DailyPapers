---
type: concept
aliases: [Ctrl-World]
category: Architecture
tags: [world-model, video-generation, latent-action, robot-manipulation]
created: 2026-06-04
---

# Ctrl-World

基于隐变量动作（Latent Action）条件的机器人操作视频世界模型，参数量 1.5B，在 DROID 数据集上训练。

## 核心特点

- **条件类型**: 隐变量动作（Latent Action）
- **参数量**: 1.5B
- **训练数据**: DROID（Franka Panda）
- **局限**: 跨形态泛化差，动作跟随精度不足

## 对比

与 [[OSCAR]] 相比：OSCAR（骨骼条件，2B）在 PSNR 上领先 5.18 dB（24.24 vs 19.06），且支持跨形态泛化。

## 相关概念

- [[Action-Conditioned Video World Model]]: 所属任务类别
- [[Diffusion Transformer]]: 生成架构
