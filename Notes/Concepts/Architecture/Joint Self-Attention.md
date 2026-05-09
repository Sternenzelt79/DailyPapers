---
concept: Joint Self-Attention
category: Architecture
tags: [transformer, attention, multi-modal]
created: 2026-05-09
---

# Joint Self-Attention

## 定义

在多流 Transformer 中，先对每条 stream（模态）独立做归一化和 QKV 投影，得到 $Q_i, K_i, V_i$；再把所有 stream 的 token 拼接后做一次自注意力：

$$
\text{Attn}\big( [Q_1; \dots; Q_n], [K_1; \dots; K_n], [V_1; \dots; V_n] \big)
$$

输出再按原 stream 边界拆回。

## 与方案对比

| 方案 | 模态参数 | 注意力 |
|------|----------|--------|
| 完全共享 | 共享 | 单次 |
| Cross-attention | 部分共享 | 多次跨模态 |
| **Joint Self-Attention** | 独立 | 单次 concat |

## 代表工作

- [[FLUX]] / [[SD3]]: text-image joint attention。
- [[Multi-Stream Action Transformer]] / [[RLDX-1]]: VLA 中的应用。

## 相关概念

- [[Multi-Stream Action Transformer]]
- [[Cross-Attention]]
