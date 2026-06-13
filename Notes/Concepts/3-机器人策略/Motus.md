---
type: concept
aliases: [Motus WAM]
---

# Motus

## 定义
一种 8B 参数的 World-Action Model，在 RoboTwin 2.0 仿真基准上达到 88.7%（Clean）的 SOTA 成功率，但推理延迟高达 3215ms/chunk。

## 核心要点
1. 8B 参数，是当前仿真基准上性能最强的 WAM
2. 每 chunk 推理延迟 3215ms，不适合实时闭环部署
3. Efficient-WAM 以 1/8 参数量和 32× 更低延迟接近其性能

## 代表工作
- [[Efficient-WAM]]: 以 Motus 为主要对比基线，验证效率优化的有效性

## 相关概念
- [[World-Action Model]]
- [[RoboTwin]]
