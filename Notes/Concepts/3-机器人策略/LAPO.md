---
type: concept
aliases: [LAPO, Latent Action Policy]
---

# LAPO

## 定义
LAPO（Latent Action Policy Optimization）学习连续的 latent action 表示，将视频理解与策略学习解耦，通过在 latent action 空间做规划来实现 model-based control。

## 核心要点
1. 从视频中学习连续 latent action（非离散 VQ）
2. latent action 用于 world model 动态预测
3. 结合 MPC 在 latent 空间做规划

## 代表工作
- LAPO (2023), Latent Action Policies for Offline RL

## 相关概念
- [[LAPA]]
- [[CLAW]]
- [[AdaWorld]]
