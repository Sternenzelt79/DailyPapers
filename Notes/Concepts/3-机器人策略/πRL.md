---
type: concept
aliases: [piRL, pi-RL, πRL]
---

# πRL

## 定义

πRL 是一种针对 VLA（Vision-Language-Action）模型的在线强化学习方法，通过固定视觉表示、仅优化动作策略（Actor-only RL）来提升机器人操作成功率。

## 核心要点

1. **Actor-only 优化**：保持视觉编码器和世界模型表示固定，只对动作头进行 RL 训练
2. **短时程有效**：在短时程任务（如 LIBERO-Object）上相对 SFT 基线有明显提升
3. **长时程局限**：在长时程多步骤任务（如 RLBench Water Plants）上几乎无提升，体现了 Actor-only 范式的根本限制
4. **代表性 baseline**：成为验证世界模型联合优化必要性的关键对比方法

## 实验数据

| 任务 | Base | πRL |
|------|------|-----|
| LIBERO-Object | 68% | 78% |
| RLBench Water Plants | 19% | 18% |

## 代表工作

- [[WAM-RL]]: 对比方法，揭示 Actor-only RL 在长时程任务上的失效

## 相关概念

- [[Reinforcement Learning]]
- [[Policy Gradient]]
- [[World-Action Model]]
- [[WAM-RL]]
- [[VLA]]
