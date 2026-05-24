---
type: concept
aliases: [Spatial Memory, Out-of-Vision Manipulation]
---

# SOMA

## 定义
Spatial Memory for Out-of-Vision Manipulation in VLA models. 为 VLA 添加持久空间记忆，使机器人能够操作当前不在摄像头视野内的目标物体。

## 核心要点
1. 从多帧历史观测构建 3D 空间记忆，借助几何估计（VGGT）
2. 物体离开视野后持久保存其位置信息供策略查询
3. 输入：多帧 RGBD + 自然语言指令 → 输出：action chunk

## 代表工作
- [[SOMA]]: Pengteng Li et al., 2026 (arXiv 2605.22283)

## 相关概念
- [[SpatialVLA]]
- [[MemoryVLA]]
- [[VLA]]
