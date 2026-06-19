---
type: concept
aliases: [LingBot VA, LingBot-VA]
---

# LingBot-VA

## 定义

一种采用**全历史 KV Cache** 的世界动作模型（WAM），通过保留完整轨迹历史的 KV Cache 实现强大的非马尔可夫推理能力，但代价是推理时显存和延迟随轨迹长度线性增长。

## 核心要点

1. **全历史保留**: KV Cache 包含整条轨迹的所有帧，上下文完整无损
2. **推理瓶颈**: $O(NL)$ 复杂度导致长轨迹下显存爆炸、延迟不可接受
3. **RMBench 强基线**: 在 9 任务上平均 78.2% 成功率，是记忆依赖任务的有力基线

## 代表工作

- 作为 [[MemoryWAM]] 的主要对比基线，后者在效率大幅提升的同时成功率超越（83.0% vs 78.2%）

## 相关概念

- [[World Action Model]]: LingBot-VA 所属的模型类别
- [[KV Cache]]: LingBot-VA 的全历史存储机制
- [[Hybrid Memory]]: MemoryWAM 用以替代全历史 KV Cache 的高效方案
