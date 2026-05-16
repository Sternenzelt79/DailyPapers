---
type: concept
aliases: [Predictive Memory, PrediMem framework]
---

# PrediMem

## 定义
基于预测机制的机器人记忆框架，用于增强 VLA 在部分可观测、长时程任务中的记忆能力，由 RoboMemArena 论文提出。

## 核心要点
1. 通过预测未来状态来形成和维护记忆，而非单纯存储历史观测
2. 与 MemoryVLA 等方法相比，侧重预测驱动的记忆更新
3. 在 RoboMemArena benchmark 上评估，覆盖 IHMB 和 HiF 等记忆依赖任务

## 代表工作
- [[RoboMemArena]]: Lei et al., 2025

## 相关概念
- [[VLA]]
- [[Memory Module]]
- [[World Model]]
