---
type: concept
aliases: [3DFlowAction, 3D Flow Action]
---

# 3DFlowAction

## 定义

3DFlowAction 是一种直接预测 3D 动作流（action flow）来引导机器人操作的方法，将 3D 运动场与动作生成结合。

## 核心要点

1. **3D 流预测**: 在 3D 空间中预测每个关键点的运动向量
2. **推理较慢**: 3.38s 推理时间（μ₀ 的约 11.7×）
3. **参数量**: 2.04B 参数

## 代表工作

- [[mu0]]: 对比基线，μ₀ 在 3D top5-DTW（T=8）上（0.127）远优于 3DFlowAction（0.529），且速度快 11.7×

## 相关概念

- [[TraceGen]]
- [[Track2Act]]
- [[Dream2Flow]]
