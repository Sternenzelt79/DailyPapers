---
type: concept
aliases: [Dreamer, DreamerV1, DreamerV2, RSSM]
---

# Dreamer

## 定义
基于循环状态空间模型 (RSSM) 的像素重建型世界模型 + actor-critic 在 latent imagination 中训练 policy。Hafner et al., 2019/2020。

## 核心要点
1. 显式重建像素 + reward + discount，端到端训练；策略在 imagined rollout 中做策略梯度。
2. 优点：对 reward 可知任务非常 sample-efficient；缺点：容量浪费在像素细节。
3. 与 [[LeWM]] 等 JEPA 路线对比：后者只在 latent 上预测、不重建像素。

## 代表工作
- Dreamer (Hafner 2020)
- [[DreamerV3]] (Hafner 2023)
- TD-MPC / TD-MPC2

## 相关概念
- [[World Model]]
- [[RSSM]]
- [[Actor-Critic]]
