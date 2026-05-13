---
type: concept
aliases: [LaRA-VLA, Latent Reasoning Action VLA]
---

# LaRA-VLA

## 定义

将多模态链式推理（multimodal chain-of-thought）内化为潜在表示的 VLA 辅助训练方法，通过课程化训练让模型在隐空间中完成推理，推理时无需额外前向传播。

## 核心要点

1. 将显式 CoT token 替换为潜在推理表示，避免推理时的额外 token 生成开销
2. 采用课程化训练策略逐步内化推理过程
3. 可将推理延迟降低最多 90%，同时保持辅助训练带来的性能增益

## 代表工作

- [[CapVector]]: 将 LaRA-VLA 的 capability vector 提取后迁移，无需在推理时运行额外模块

## 相关概念

- [[VLA]]
- [[辅助目标微调]]
- [[监督微调]]
