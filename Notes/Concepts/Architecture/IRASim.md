---
type: concept
aliases: [IRASim]
category: Architecture
tags: [world-model, video-generation, latent-action, robot-manipulation]
created: 2026-06-04
---

# IRASim

基于隐变量动作（Latent Action）条件的早期机器人操作视频世界模型，参数量 0.7B，在 RT-1、Bridge、Language-Table、RoboNet 上训练。

## 核心特点

- **条件类型**: 隐变量动作（Latent Action）
- **参数量**: 0.7B
- **训练数据**: RT-1, Bridge, Language-Table, RoboNet
- **局限**: 视频质量差（PSNR 6.48，SSIM 0.088），动作跟随极不精确

## 对比

在 [[OSCAR]] 论文的评估中表现最差（FVD 411.42，FID 394.10），说明仅靠隐变量动作条件难以实现精确动作跟随。

## 相关概念

- [[Action-Conditioned Video World Model]]: 所属任务类别
- [[Ctrl-World]]: 同为隐变量动作方法的改进版
