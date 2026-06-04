---
type: concept
aliases: [NavRL, Navigation with RL]
---

# NavRL

## 定义
NavRL 是基于强化学习的视觉导航方法，在仿真环境中训练视觉策略，实现 sim-to-real 迁移到真实无人机或地面机器人。

## 核心要点
1. 直接端到端 RL 训练，输入深度图/RGB，输出控制命令
2. 大规模随机化仿真环境，提高 sim-to-real 泛化
3. 比传统 SLAM+规划 更轻量，适合低算力平台

## 相关概念
- [[MAD]]
- [[DreamerV3]]
- [[PPO]]
