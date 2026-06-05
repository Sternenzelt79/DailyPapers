---
concept: Action Chunking
category: VLA
tags: [vla, action, control]
created: 2026-05-09
---

# Action Chunking

## 定义

Action Chunking 指策略一次预测**未来 $H$ 步动作序列** $a_{t:t+H}$ 而不是单步动作。执行时可以全部执行、按 receding-horizon 重新规划，或与 [[Temporal Ensembling]] 结合。

## 优点

- 缓解 [[Compounding Error]]：高频再规划下分布漂移更小。
- 提高 throughput：一次推理多步。
- 便于与 [[Flow Matching]] / [[Diffusion Model]] 等生成式 head 结合（自然输出固定长度序列）。

## 典型 horizon

| 方法 | $H$ |
|------|-----|
| ACT | 100 |
| Pi0 | 50 |
| RLDX-1 (ALLEX) | 40 |
| RLDX-1 (FR3) | 16 |

## 代表工作

- [[ACT]] (2023): 提出 action chunking 概念。
- [[Pi0]], [[RLDX-1]]: 与 flow matching 结合。
- [[PiL-World]] (2026): 将 action chunk 作为世界模型视觉条件（chunk-wise 预测），驱动 VLA 闭合循环评估。

## 相关概念

- [[Temporal Ensembling]]
- [[Vision-Language-Action Model]]
- [[Flow Matching]]
