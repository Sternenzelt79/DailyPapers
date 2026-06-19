---
type: concept
aliases: [RMBench, Robot Memory Benchmark]
---

# RMBench

## 定义

一个专为评估机器人策略的**非马尔可夫记忆能力**而设计的双臂操控基准，包含 9 个具有不同记忆依赖复杂度的任务，要求智能体依赖历史观测作出正确决策。

## 核心要点

1. **记忆依赖设计**: 所有任务均要求策略维护跨时间步的信息（如物体位置、任务顺序）
2. **多复杂度分级**: 9 个任务覆盖从简单到复杂的不同记忆需求层级
3. **双臂操控场景**: 基于双臂机器人平台的仿真评测

## 典型任务

- **Shell Game**: 跟踪物体在杯子间的移动轨迹
- **Cover Blocks**: 记住遮盖操作的顺序
- **Press Button**: 根据历史指示按下对应按钮
- Rearrange Blocks / Swap Blocks / Swap T / Battery Try 等

## 代表工作

- [[MemoryWAM]]: 在 RMBench 9 个任务上平均成功率 83.0%，超越 LingBot-VA（78.2%）和 FastWAM（5.9%）

## 相关概念

- [[World Action Model]]: 在 RMBench 上评测的主要模型类别
- [[Non-Markovian Policy|非马尔可夫策略]]: RMBench 专门评测此类能力
