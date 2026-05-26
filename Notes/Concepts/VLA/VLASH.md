---
type: concept
aliases: [VLA Staleness Handling]
---

# VLASH

## 定义
VLASH 是针对 VLA 异步推理延迟问题的早期方法，处理机器人在执行旧动作块期间新观测已过时（stale action）的问题。

## 核心要点
1. VLA 异步推理：模型计算新动作时机器人还在执行旧动作，导致 stale action 问题
2. VLASH 是该问题的 baseline 方法，被后续工作（如 DEFLECT）用作对比

## 代表工作
- [[DEFLECT]]: 提出 Flow Matching 似然估计替代 VLASH

## 相关概念
- [[Conditional Flow Matching]]
- [[VLA]]
