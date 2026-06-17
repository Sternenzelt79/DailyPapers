---
type: concept
aliases: [Self-Recovery, Visual Self-Recovery, 视觉自恢复]
---

# Self-Recovery

## 定义
让多模态模型在推理时先对损坏的视觉输入进行自我修复（图像恢复），再进行下游理解任务，以提升对真实世界视觉腐蚀的鲁棒性。

## 核心要点
1. 在同一模型内集成视觉恢复和语义理解两个能力
2. 训练分两阶段：SFT 学会恢复，再用 RL 对齐更高视觉质量
3. 与外部恢复模块不同，无需额外网络，内化于模型权重

## 代表工作
- [[Robust-U1]]: 提出 Self-Recovery 框架，用于 MLLM 的鲁棒视觉理解

## 相关概念
- [[分布外泛化]]
- [[BAGEL]]
- [[TeCoA]]
