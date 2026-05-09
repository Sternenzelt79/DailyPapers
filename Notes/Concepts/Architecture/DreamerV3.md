---
type: concept
aliases: [DreamerV3]
---

# DreamerV3

## 定义
[[Dreamer]] 系列的第三代，提出 symlog 预测、free-bits、tunable-fixed 超参组合，实现"一套超参跑 150+ 任务"的通用 world model。Hafner et al., 2023。

## 核心要点
1. 仍是像素重建型 RSSM；强调通用性（Atari, DMC, Crafter, Minecraft）。
2. 用 symlog 处理多尺度 reward / value。
3. 是当前像素重建路线的代表，对照 JEPA 路线（[[LeWM]]）讨论。

## 代表工作
- DreamerV3 (Hafner 2023)
- [[Dreamer]]

## 相关概念
- [[World Model]]
- [[Dreamer]]
