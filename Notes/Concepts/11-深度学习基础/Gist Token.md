---
type: concept
aliases: [Gist Tokens, 摘要Token, 压缩Token]
---

# Gist Token

## 定义

一组可学习的参数化 Token，用于将较长的输入序列（如一帧的全部视觉 Token）压缩为更紧凑的表示，在保留关键语义的同时大幅减少存储/计算开销。

## 核心要点

1. **可学习压缩**: Gist Token 的参数在训练中被优化，学会聚合输入帧的关键语义特征
2. **持久性**: 在自回归推理中，Gist Token 对应的 KV Cache 永久保留，不随滑动窗口驱逐
3. **线性压缩**: 将每帧 $L$ 个 Token 压缩为 $M$ 个（$M \ll L$），压缩比 $d = L/M$
4. **与 Cross-Attention 结合**: 通常通过交叉注意力机制让 Gist Token 查询对应帧的视觉 Token

## 数学形式

$$
|\mathcal{C}^v_\text{gist}| = O(N \cdot M) = O\!\left(\frac{NL}{d}\right)
$$

每帧 Gist Token 数 $M \ll L$（如 MemoryWAM 中 $M=8$，$L=120$，$d=15$）。

## 代表工作

- [[MemoryWAM]]: 每帧 8 个 Gist Token，将长程历史压缩 15 倍后持久存储于 KV Cache

## 相关概念

- [[KV Cache]]: Gist Token 压缩结果以 KV 形式持久化存储
- [[Hybrid Memory]]: Gist Token 是混合记忆长程组件的实现
- [[Cross-Attention|交叉注意力]]: Gist Token 通过交叉注意力聚合帧信息
