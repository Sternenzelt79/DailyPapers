---
type: concept
category: Training
tags: [reinforcement-learning, vlm]
created: 2026-05-09
---

# VLM Critic

把 VLM 当 RL critic 用：让 VLM 把价值估计写成 unnormalized 整数 token（如 0–100），借用既有 LM head 的 softmax 输出概率分布，期望即 value。优点是无需训练新回归头、可以零成本利用 VLM 的视觉-语言理解。

## 代表工作

- [[RECAP]] / [[RLDX-1]]: 在 post-training RL 中用此机制评估机器人任务进度。
