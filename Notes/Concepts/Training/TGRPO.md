---
type: concept
aliases: [Trajectory GRPO, Trajectory-level Group Relative Policy Optimization]
---

# TGRPO

## 定义
TGRPO 是 [[GRPO]] 的轨迹级扩展，对完整动作序列而非单步动作进行组相对策略优化，用于 VLA 的强化学习微调。

## 核心要点
1. 标准 GRPO 对单步 token 做组相对打分；TGRPO 以完整轨迹为单位计算 advantage
2. 轨迹级奖励减少了单步信用分配（credit assignment）的噪声
3. 主要应用于 VLA 的 RLVR 训练场景

## 代表工作
- [[PAPO-VLA]]: 将 TGRPO 作为 baseline 并提出 Planning-Aware 扩展

## 相关概念
- [[GRPO]]
- [[FlowGRPO]]
- [[GRAPE]]
