---
type: concept
aliases: [VGGT, Visual Geometry Grounded Transformer]
---

# VGGT

## 定义
单目 3D 几何估计 Transformer，从单帧或多帧 RGB 图像预测相机位姿、深度图和 3D 点云，在 SOMA 中用于构建空间记忆。

## 核心要点
1. 端到端预测 3D 几何（深度 + 相机位姿）
2. 无需多视角输入，单目可用
3. 在 SOMA 中为空间记忆提供 3D 位置信息

## 相关概念
- [[SOMA]]
- [[相机位姿估计]]
