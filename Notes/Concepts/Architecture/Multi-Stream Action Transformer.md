---
concept: Multi-Stream Action Transformer
aliases: [MSAT]
category: Architecture
tags: [vla, transformer, multi-modal]
created: 2026-05-09
---

# Multi-Stream Action Transformer (MSAT)

## 定义

MSAT 是一种为 VLA（Vision-Language-Action）任务设计的 Transformer 变体。它为每种模态（cognition / action / physics）分配独立的 stream，每条 stream 拥有自己的 RMSNorm 和 QKV 投影，但在每层内通过 **concat → joint self-attention** 让所有 stream token 互相交互。

## 与单流 Transformer 的区别

| 维度 | 单流 (e.g. π₀.₅) | 多流 MSAT |
|------|-------------------|-----------|
| 模态归一化 | 共享 | 独立 |
| QKV 投影 | 共享权重 | 每流独立 |
| 注意力 | 一次自注意力 | concat 后联合自注意力 |
| 模态干扰 | 高 | 低 |

## Block 形态

- **Double-stream**: cognition + action（早期层）。
- **Triple-stream**: 加入 physics（中段层）。
- **Merged**: cognition 与 action 合并，physics 可选保留（末段）。

## 代表工作

- [[RLDX-1]] (2026): 首次提出并在 ALLEX 人形机器人上达到 86.8% 成功率。

## 相关概念

- [[Joint Self-Attention]]
- [[Vision-Language-Action Model]]
- [[Flow Matching]]
