---
type: concept
category: Architecture
tags: [position-embedding, transformer]
created: 2026-05-09
---

# RoPE (Rotary Position Embedding)

通过把 query / key 在复数平面上按位置旋转的方式注入相对位置信息的位置编码。已成为 LLM 的事实标准。

## 代表工作

- LLaMA / Qwen 等主流 LLM
- [[RLDX-1]]: 在 MSAT 的 action stream 上施加 RoPE 编码动作时序结构。
