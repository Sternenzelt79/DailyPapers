---
type: concept
category: Architecture
tags: [multi-modal, masking]
created: 2026-05-09
---

# Stream Masking

多模态多流架构中的开关机制：当某条模态流（如物理感知）在当前硬件 / 平台不可用时，把整条 stream 的 token 屏蔽，使模型回退到剩余流上仍可工作。

## 代表工作

- [[RLDX-1]]: 在没有 torque/tactile 传感器的平台上把 [[Physics Stream]] 整条 mask 掉，保证训练与推理可在异构硬件间共享。
