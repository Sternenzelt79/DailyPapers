---
type: concept
category: Architecture
tags: [diffusion, transformer, multi-modal]
created: 2026-05-09
---

# MM-DiT (Multi-Modal Diffusion Transformer)

由 Stable Diffusion 3 提出的双流 Diffusion Transformer：文本与图像各占一条独立 stream，在每个 block 内通过 joint self-attention 跨模态交互；模态自身的 LayerNorm / FFN 不共享，从而保留模态内统计。

## 代表工作

- Stable Diffusion 3
- [[RLDX-1]] 的 [[Multi-Stream Action Transformer|MSAT]] 把 MM-DiT 推广到 cognition / action / physics 三流。
