---
concept: Memory Module
category: Architecture
tags: [memory, vla, long-context]
created: 2026-05-09
---

# Memory Module (VLA 长期记忆)

## 定义

VLA 中用于跨多步任务保持状态的轻量级模块。从 LLM 中间层抽取 cognition feature，按固定间隔采样维护 FIFO 队列，再用一个小型因果 Transformer 聚合得到 memory token。

## 典型实现 ([[RLDX-1]])

- 抽取层：LLM 第 4 层后。
- 队列长度：$n_{mem} = 3$。
- 采样间隔：$H + 1$ 步（与 action chunk 对齐）。
- 聚合方式：[[Causal Attention]] 轻量 Transformer。

## 解决的任务类型

- Object-in-Box：记住"目标已放入哪个盒子"。
- 多阶段烹饪 / 装配。
- 任何需要跨秒级时间尺度的状态保持。

## 代表工作

- [[RLDX-1]]: Object-in-Box 任务从 ~30% 提升到 91.7%。

## 相关概念

- [[Long-Context]]
- [[Causal Attention]]
- [[Compressive Transformer]]
