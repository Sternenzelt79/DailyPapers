---
concept: World Model
category: VLA
tags: [world-model, embodied-ai, prediction]
created: 2026-05-09
---

# World Model

## 定义

智能体对环境演化规律的内部表示：给定当前观测/状态与动作，能预测下一步观测、状态或某种压缩潜变量。在 [[VLA]] / 机器人语境下，世界模型常以隐式（latent dynamics）或显式（深度、点云、占据网格）形式存在，作为动作生成的条件。

## 形式分类

| 类型 | 表示 | 代表 |
|------|------|------|
| 像素级 | 直接预测 RGB | Dreamer 系列 |
| 隐空间 | 预测 latent | DreamerV3, [[JEPA]] |
| 几何级 | 深度 / 占据网格 / 3DGS | [[MolmoAct2-Think]], [[3D Gaussian Splatting]] |
| token 化 | 离散 token 序列 | [[Adaptive Depth Reasoning|自适应深度]] |

## 在 VLA 中的作用

1. 提供动作之前的"预演 / 预测"线索。
2. 提升长程任务规划与泛化能力。
3. 作为辅助监督信号（深度预测 / 未来帧预测）。

## 代表工作

- [[MolmoAct2]] / [[MolmoAct2-Think]]: 把深度 token 网格作为隐式世界模型注入动作专家。
- [[LeWM]]: latent world model 路线。
- [[JEPA]]: 联合嵌入预测架构。

## 相关概念

- [[JEPA]]
- [[Adaptive Depth Reasoning]]
- [[3D Gaussian Splatting]]
- [[Diffusion Model]]
