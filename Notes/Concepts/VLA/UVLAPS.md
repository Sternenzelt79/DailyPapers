---
type: concept
aliases: [UVLAPS, VLA-guided Planning]
---

# UVLAPS

## 定义
VLA-guided Monte Carlo Tree Search 规划方法，使用预训练 VLA 策略作为 MCTS 的动作先验，通过树搜索在测试时改善 VLA 的长时序任务执行效果；V-VLAPS 是在此基础上引入显式 value 估计的改进版本。

## 核心要点
1. 用预训练 VLA 策略替代 MCTS 的随机 rollout，提升搜索效率
2. 基于 PUCT（Upper Confidence bounds for Trees）节点选择策略
3. 显著改善分布偏移和长时序场景下的任务成功率
4. 主要计算开销是树搜索展开，难以直接用于实时控制

## 代表工作
- [[V-VLAPS]]: 在 UVLAPS 基础上引入显式 value network，改进 PUCT 节点选择

## 相关概念
- [[V-VLAPS]]
- [[RT-2]]
- [[OpenVLA]]
