---
type: concept
aliases: [MaskNet]
---

# MaskNet

## 定义
MaskNet 是一种用于机器人操作的目标感知分割-动作联合模型，通过显式分割任务相关物体来减少视觉干扰，提升操作策略的鲁棒性。

## 核心要点
1. 联合学习目标分割和动作预测
2. 将分割 mask 作为中间表示指导动作
3. 减少 VLM/VLA 的视觉幻觉问题

## 代表工作
- MaskNet (2024), Mask-Guided Robot Manipulation

## 相关概念
- [[SceneDiver]]
- [[OvSGTR]]
