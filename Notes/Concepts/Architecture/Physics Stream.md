---
concept: Physics Stream
category: Architecture
tags: [physics, multi-modal, vla]
created: 2026-05-09
---

# Physics Stream

## 定义

Physics Stream 是一条与 action stream 并行的 Transformer 流，专门预测**未来物理传感读数**（接触力 / 触觉 / IMU 等），用 [[Flow Matching]] 目标联合训练。

## 与 Action Stream 的关系

- 共享 cognition 上下文。
- 独立的 RMSNorm / QKV 投影。
- 通过 [[Joint Self-Attention]] 与 action token 交互。
- 训练时 `L_total = L_action + λ_p * L_physics`。

## 作用

- 提供隐式 [[World Model]]：模型在生成动作时同时预测物理后果。
- 灵巧抓取/插入类任务收益最大。

## 代表工作

- [[RLDX-1]]: 引入 physics stream 后灵巧任务平均提升 ~8 个百分点。

## 相关概念

- [[Multi-Stream Action Transformer]]
- [[Flow Matching]]
- [[World Model]]
