---
type: concept
aliases: [VSI, Visual Spatial Intelligence, VSI-Bench]
---

# VSI

## 定义
评估 MLLM 视觉空间智能的 benchmark，专注于 3D 场景扫描视频中的空间推理问题，包括物体计数、距离估计、尺寸判断等。

## 核心要点
1. 使用真实室内场景扫描（如 ScanNet）作为输入视频
2. 问题涵盖：绝对/相对距离、物体大小、方向判断
3. 测试模型在真实 3D 场景视频中的定量空间推理
4. 被 [[OVO-S-Bench]] 用作对比——VSI 测完整视频而非 streaming 流

## 相关概念
- [[MMSI]]
- [[OVO-S-Bench]]
