---
type: concept
category: Architecture
tags: [vla, vlm, token]
created: 2026-05-09
---

# Cognition Token

VLM 与下游 action head 之间的可学习接口 token，用于把视觉-语言上下文聚合成定长、密集的表示。

类似 [[Q-Former]] 或 BLIP-2 的 learnable query，但聚焦 action-relevant 的语义抽取。

## 代表工作

- [[RLDX-1]]: 64 个 learnable cognition tokens，从 [[Qwen3-VL]] 中聚合视频与指令，喂给 [[Multi-Stream Action Transformer|MSAT]] 的 cognition stream，并作为 [[Memory Module]] 队列的存储单元。
