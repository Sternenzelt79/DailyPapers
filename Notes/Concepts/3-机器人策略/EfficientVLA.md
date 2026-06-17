---
type: concept
aliases: [Efficient VLA, VLA Compression]
---

# EfficientVLA

## 定义
对 VLA 模型进行结构化剪枝/压缩的方法，在保持操控性能的同时大幅降低推理计算量，面向边缘端部署。

## 核心要点
1. 结构化剪枝减少 VLA 参数量
2. 在 LIBERO 等基准上验证压缩后性能
3. RLRC 等方法用其作为压缩基线

## 代表工作
- [[RLRC]]: 在 EfficientVLA 压缩后用 RL 做性能恢复

## 相关概念
- [[VLA]]
- [[FLAP]]
- [[LightVLA]]
