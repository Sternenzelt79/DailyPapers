---
type: concept
aliases: [Segment Anything Model 2, SAM 2, 视频分割模型]
---

# SAM2（Segment Anything Model 2）

## 定义

Meta 发布的视频目标分割基础模型，能够在视频流中实时追踪并分割任意指定的目标区域，支持点、框、mask 等多种交互提示。

## 核心要点

1. 基于 Hiera 图像编码器 + 流式记忆机制，实现帧间目标追踪
2. 支持零样本分割，无需针对特定物体类别训练
3. 推理速度快（约 44 FPS on A100），适合在线应用
4. 在 SA-V 数据集上训练，覆盖多样化视频场景

## 代表工作

- [[GRAIL]]: 用 SAM2 预测物体 mask，与 FoundationPose 重投影 mask 对比计算追踪误差 $e_M$，过滤低质量 4D HOI 重建序列

## 相关概念

- [[FoundationPose]]
- [[HOI]]
