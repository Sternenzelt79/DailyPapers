---
type: concept
aliases: [混合记忆, 分级记忆, Hybrid Memory Cache]
---

# Hybrid Memory

## 定义

一种将不同时间粒度的历史信息以不同压缩程度分别存储的记忆架构，通常用于 Transformer 的 KV Cache 管理，以在长序列推理中平衡效率和上下文完整性。

## 核心要点

1. **分级设计**: 近期帧完整保留（高精度短程）+ 事件边界锚帧保留（任务语义）+ 中间历史压缩存储（低开销长程）
2. **复杂度优化**: 相比全历史 KV Cache 的 $O(NL)$，混合记忆将长程部分压缩为 $O(NM)$（$M \ll L$），总复杂度降至 $O(NL/d)$
3. **无损关键信息**: 通过选择性保留策略，使最重要的上下文（近期 + 事件边界）不被压缩

## 数学形式

$$
\mathcal{C}^v_{\leq t} = \mathcal{C}^v_\text{short} \cup \mathcal{C}^v_\text{anchor} \cup \mathcal{C}^v_\text{gist}
$$

其中 $|\mathcal{C}^v_\text{gist}| = O(N \cdot M) = O(NL/d)$，$d = L/M$ 为压缩比。

## 代表工作

- [[MemoryWAM]]: 提出用于机器人操控 WAM 的三组件混合记忆（$N_\text{recent}=4$，$N_\text{init}=2$，$M_v=8$，压缩比 15×）

## 相关概念

- [[KV Cache]]: 混合记忆的底层实现机制
- [[Gist Token]]: 用于长程历史压缩的可学习 Token
- [[World Action Model]]: 混合记忆的应用场景
