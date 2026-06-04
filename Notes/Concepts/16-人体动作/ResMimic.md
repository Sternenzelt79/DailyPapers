---
type: concept
aliases: [ResMimic, 残差模仿学习]
---

# ResMimic

## 定义

一种基于残差模仿学习的仿人机器人运动追踪方法，在预训练运动先验基础上通过残差策略学习特定操作技能。

## 核心要点

1. 使用残差策略在基础控制器之上叠加操作相关动作
2. 依赖预录制的人类示范视频作为参考运动
3. 在 GRAIL 评测中作为主要对比基线：SR 49.2% vs GRAIL 81.4%

## 代表工作

- [[GRAIL]]: ResMimic 作为对比基线，GRAIL 在 SR 上以 81.4% vs 49.2% 显著超越

## 相关概念

- [[Loco-Manipulation]]
- [[SONIC]]
- [[HDMI]]
