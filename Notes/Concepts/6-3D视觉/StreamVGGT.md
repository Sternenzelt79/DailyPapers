---
type: concept
aliases: [Streaming VGGT, 实时VGGT]
---

# StreamVGGT

## 定义
[[VGGT]] 的流式（causal）变体，支持实时、在线的3D场景重建，可在视频流输入下逐帧更新3D几何表示。

## 核心要点
1. 将 VGGT 的双向注意力改为因果注意力，使其支持流式推理
2. 每帧输入后立即输出更新后的3D结构（点云/深度图/相机位姿）
3. 是 [[PointAction]] 的关键依赖：实时3D动作表示需要流式3D重建

## 代表工作
- [[PointAction]]: 用 StreamVGGT 做实时3D点云动作表示，用于机器人操作

## 相关概念
- [[VGGT]] — 基础模型
- [[PointAction]] — 下游应用
- [[相机位姿估计]] — 相关任务
