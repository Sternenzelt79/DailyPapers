---
type: concept
aliases: [Dream2Flow]
---

# Dream2Flow

## 定义

Dream2Flow 是一种基于视频扩散模型生成未来帧，再从中提取 3D 运动流来引导机器人操作的方法。

## 核心要点

1. **视频扩散 + 流提取**: 先生成未来视频，再提取 3D 光流作为操作引导
2. **推理极慢**: 106.8s 推理时间（μ₀ 的约 368×），难以实际部署
3. **大参数量**: 11.3B 参数

## 代表工作

- [[mu0]]: 对比基线，μ₀ 在 3D top5-DTW（T=8）上（0.127）优于 Dream2Flow（0.198），且速度快 368×

## 相关概念

- [[3DFlowAction]]
- [[TraceGen]]
- [[Conditional Flow Matching]]
