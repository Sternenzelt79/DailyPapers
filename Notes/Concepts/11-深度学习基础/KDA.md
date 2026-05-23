---
type: concept
aliases: [Kimi Delta Attention, Kimi Delta Rule]
---

# KDA

## 定义
Kimi Delta Attention，一种线性注意力机制变体，通过 delta-rule（当前读取值减去新写入值）实现对固定大小循环状态的精细编辑，解决传统线性注意力"写新值会覆盖旧关联"的问题。

## 数学形式
Delta-rule 更新：
$$\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{k}_t^\top (\mathbf{v}_t - \mathbf{k}_t \mathbf{S}_{t-1})$$

Gated DeltaNet-2 改进版（解耦 erase/write）：
$$\bm{G}_r = \sum_{i=1}^r \bm{g}_i, \quad \bm{\gamma}_r = \exp(\bm{G}_r)$$

## 核心要点
1. 固定大小循环状态 → 推理时 $O(1)$ 内存，序列混合 $O(n)$ 时间
2. Delta-rule 解决"如何编辑压缩记忆而不打乱已有关联"的问题
3. Gated DeltaNet-2 将 erase gate 和 write path 解耦，提升长上下文任务性能（NIAH-1/2/3）

## 代表工作
- [[GatedDeltaNet-2]]: NVIDIA 改进版，在 [[RULER]]、[[ARC]] 等 benchmark 上超过原版 KDA

## 相关概念
- [[Transformer]]
- [[滑动窗口注意力]]
- [[多头自注意力]]
