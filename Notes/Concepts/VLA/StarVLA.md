---
type: concept
aliases: [StarVLA]
---

# StarVLA

## 定义

模块化统一 VLA 框架，通过 backbone-action-head 抽象设计支持范式无关（paradigm-agnostic）训练策略，提供标准化评估接口，兼容自回归和流匹配两种范式。

## 核心要点

1. 模块化架构：骨干网络、动作头分离，便于替换和组合
2. 范式无关训练：同时支持自回归和流匹配两种动作预测范式
3. 标准化评估接口，便于跨方法比较

## 代表工作

- [[CapVector]]: 以 StarVLA 为基础模型（配合 LaRA-VLA 辅助训练），验证 capability vector 的跨架构通用性

## 相关概念

- [[VLA]]
- [[辅助目标微调]]
- [[Conditional Flow Matching]]
