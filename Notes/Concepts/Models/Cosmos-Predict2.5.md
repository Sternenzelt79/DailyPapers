---
type: concept
aliases: [Cosmos-Predict2.5-2B]
category: Models
tags: [video-generation, world-model, diffusion-transformer]
created: 2026-06-04
---

# Cosmos-Predict2.5

NVIDIA Cosmos 世界模型系列中的 image-to-video 预测模型（2B 参数），使用 [[Rectified Flow]] 目标和 [[WAN 2.1 VAE]] 进行时空编解码，是 Cosmos-Predict2 的升级版本。

## 核心特点

- **架构**: 2B 参数 [[Diffusion Transformer]]
- **VAE**: WAN 2.1（时空联合编码）
- **训练目标**: [[Rectified Flow]]（整流流）
- **用途**: 机器人操作视频生成、世界模型微调基座

## 代表工作

- [[OSCAR]]: 在 Cosmos-Predict2.5-2B 上微调，添加骨骼条件注入，在单张 GH200 上实现跨形态机器人视频世界模型

## 相关概念

- [[Cosmos-Predict2]]: 前代版本
- [[Diffusion Transformer]]: 核心生成架构
- [[Rectified Flow]]: 训练目标
