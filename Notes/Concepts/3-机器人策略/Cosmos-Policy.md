---
type: concept
aliases: [Cosmos Policy, CosmosPolicy]
---

# Cosmos-Policy

## 定义

Cosmos-Policy 是 NVIDIA 提出的视频世界动作模型（WAM），将大规模视频预训练的 Cosmos 世界模型与机器人动作生成相结合，在 2D 视频空间中预测未来帧并生成控制指令。

## 核心要点

1. **基于视频 WAM 范式**: 在像素空间生成未来视频帧，缺乏显式 3D 几何理解
2. **模型规模**: 约 2B 参数
3. **推理延迟高**: 约 382ms（~3 Hz），不适合实时控制
4. **LIBERO 表现**: Orig 98.5%，Plus 82.4%（↓16.1%），摄像头鲁棒性 73.4%

## 代表工作

- [[GAM]]: 对比基线，GAM 比 Cosmos-Policy 快 55×（6.9ms vs 382.4ms）

## 相关概念

- [[World-Action Model]]
- [[时序世界建模]]
- [[Fast-WAM]]
