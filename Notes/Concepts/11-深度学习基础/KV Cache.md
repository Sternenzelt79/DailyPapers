---
type: concept
aliases: [KV Cache, KV 缓存, Key-Value Cache]
---

# KV Cache

## 定义
Transformer 自回归推理时，将已计算好的 Key 和 Value 矩阵缓存起来，避免重复计算，从而在保持质量不变的前提下大幅加速推理。

## 数学形式
对于每个注意力头，缓存所有历史 token 的 K 和 V：
$$
\text{Attn}(Q_t, [K_{1:t}], [V_{1:t}]) = \text{softmax}\left(\frac{Q_t [K_{1:t}]^\top}{\sqrt{d}}\right)[V_{1:t}]
$$

每步新增 token 时只计算新的 $Q_t, K_t, V_t$，拼接历史缓存即可。

## 核心要点
1. **节省计算**: 将二次复杂度的重复计算降为线性增量计算
2. **内存换时间**: 需要存储所有历史层的 K/V，内存占用随序列长度线性增长
3. **扩散模型中的应用**: 在多步去噪过程中缓存条件 token 的 K/V（如 Memory/History）

## 代表工作
- [[WEAVER]]: 在扩散去噪步骤间缓存 Memory 和 History token 的 K/V，实现 3× 以上推理加速

## 相关概念
- [[Transformer]]
- [[2D Transformer]]
