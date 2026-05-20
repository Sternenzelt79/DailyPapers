---
type: concept
aliases: [Trajectory-level GRPO, T-GRPO]
---

# TGRPO

## 定义

TGRPO（Trajectory-level GRPO）是将 [[GRPO]] 应用于 [[Vision-Language-Action Model|VLA]] 策略优化的方法，通过轨迹级奖励信号引导 VLA 模型的策略更新，提升机器人操作任务的成功率。

## 核心要点

1. 将 [[GRPO]] 的组相对优势估计迁移到机器人操作的连续动作空间
2. 使用轨迹级任务奖励（成功/失败）作为优化信号
3. 对轨迹内所有时间步采用均匀权重进行策略更新

## 与 PAPO-VLA 对比

TGRPO 未区分规划动作和执行动作，对所有时间步均匀加权；[[PAPO-VLA]] 在其基础上引入规划感知优势估计，LIBERO Avg. 从 0.81 提升至 0.96。

## 代表工作

- [[PAPO-VLA]]: 在 TGRPO 基础上引入因果感知规划动作权重

## 相关概念

- [[GRPO]]
- [[Vision-Language-Action Model]]
- [[PAPO-VLA]]
- [[RIPT-VLA]]
