---
type: concept
aliases: [Cosmos-Policy, CosmosPolicy]
---

# Cosmos Policy

## 定义
NVIDIA Cosmos 系列的机器人操作策略模型，兼具视频预测（场景演化）和动作生成能力，可作为 [[World-Action Model|WAM]] 使用。

## 核心要点
1. 兼具场景预测与动作生成：Cosmos-Predict 侧重纯场景预测，Cosmos Policy 经动作后训练支持动作输出
2. 在 [[WorldPilot]] 中作为冻结 WAM 使用，5步去噪生成场景演化潜变量和预期轨迹
3. 即使仅使用 Cosmos-Predict（无动作后训练）的场景先验，仍可带来 RoboCasa +8.7pt 提升

## 代表工作
- [[WorldPilot]]: 使用 Cosmos Policy 作为冻结 WAM，提供 [[Latent Steering]] 和 [[Action Steering]] 所需先验

## 相关概念
- [[Cosmos-Predict]]
- [[World-Action Model]]
- [[Diffusion Transformer]]
- [[VAE]]
