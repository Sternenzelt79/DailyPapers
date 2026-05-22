---
type: concept
aliases: [SnapFlow, Flow-Matching DPO]
---

# SnapFlow

## 定义
适配 flow-matching VLA 的偏好优化框架（DPO 变体），解决标准 DPO 无法直接应用于 ODE-based 连续动作生成过程的问题；是 CrossVLA 的核心组件。

## 核心要点
1. 把 DPO 偏好对齐推广到 flow-matching 范式
2. 在 ODE trajectory 层面定义 preference
3. 配合 CrossVLA 使用

## 相关概念
- [[Flow Matching]]
- [[CrossVLA]]
