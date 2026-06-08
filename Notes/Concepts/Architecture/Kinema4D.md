---
type: concept
aliases: [Kinema4D]
category: Architecture
tags: [world-model, video-generation, point-map, robot-manipulation, 3d-vision]
created: 2026-06-04
---

# Kinema4D

基于点图（Point Map）条件的机器人操作视频世界模型，参数量 14B，在 Robo4D-200k 数据集上训练。以大参数量换取高精度的代表方法。

## 核心特点

- **条件类型**: 点图渲染（Point Map，3D 点云投影到 2D）
- **参数量**: 14B（最大的基线方法）
- **训练数据**: Robo4D-200k
- **推理速度**: 0.089 FPS（极慢）
- **性能**: PSNR 17.68，SSIM 0.741，FVD 17.07

## 局限

尽管参数量是 OSCAR 的 7 倍，但 PSNR 仅 17.68（vs OSCAR 24.24），FPS 仅 0.089（vs OSCAR 2.214），说明点图条件存在表达过度复杂、纹理过拟合等问题。

## 对比

- [[OSCAR]]: 2B 参数骨骼条件，以 1/7 参数量取得更好性能，推理速度快 25 倍
- [[Genie Envisioner]]: 同为几何渲染条件，但参数量更小、更高效

## 相关概念

- [[Action-Conditioned Video World Model]]: 所属任务类别
- [[Diffusion Transformer]]: 生成架构
