---
type: concept
aliases: [GenieEnvisioner, Genie Envisioner]
---

# Genie-Envisioner

## 定义

Genie-Envisioner 是一个面向机器人操作的世界基础模型（World Foundation Model），在 WorldArena 和 AgiBot-World benchmark 上作为主要竞争基线。

## 核心要点

1. **WorldArena 表现**: EWMScore 68.26（排名第 2，低于 PAIWorld 的 70.67）
2. **文本条件生成优势**: 在 AgiBot-World 文本条件生成任务上 Scene Consistency（0.9231）高于 PAIWorld（0.9041）
3. **参数量**: 存在 2B 参数量的小型版本 Genie-Envisioner-Sim2.0-2B
4. **局限**: LPIPS（0.3345 vs 0.1844）和 MEt3R（15.75 vs 14.20）显著差于 PAIWorld，说明感知质量和 3D 一致性较弱

## 代表工作

- [[PAIWorld]]: 将 Genie-Envisioner 作为主要对比 baseline

## 相关概念

- [[WorldArena]]
- [[AgiBot-World]]
- [[条件视频生成]]
