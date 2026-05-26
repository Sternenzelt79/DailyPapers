---
type: concept
aliases: [Group Relative Action Policy Evaluation]
---

# GRAPE

## 定义
GRAPE 是用于 VLA 策略优化的强化学习方法，基于组相对策略评估对 VLA 动作预测进行 RL 微调，属于 GRPO 变体家族。

## 核心要点
1. 以动作组为单位计算相对 advantage，避免绝对 reward 的量纲问题
2. 在 VLA 闭环操作任务中用于细粒度动作质量评估
3. 与 TGRPO 同属轨迹/动作级 GRPO 变体，侧重点不同

## 代表工作
- [[PAPO-VLA]]: 将 GRAPE 列为对比 baseline

## 相关概念
- [[GRPO]]
- [[TGRPO]]
- [[Reinforcement Learning]]
