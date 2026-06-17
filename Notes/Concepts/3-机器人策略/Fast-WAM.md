---
type: concept
aliases: [Fast WAM, FastWAM]
---

# Fast-WAM

## 定义

Fast-WAM 是一种高效的 World-Action Model，使用双向扩散 Transformer 建模物理动态，以加速推理为核心设计目标，但缺乏语言生成能力，难以完成需要高层语义推理的长时程任务。

## 核心要点

1. 基于双向扩散 Transformer（非自回归），适合并行解码
2. 参数规模约 6B，推理速度相对较快
3. 无语言生成分支，高层规划依赖外部模块
4. 在长时程任务（RMBench）中成功率显著低于具备语言推理的 WLA

## 代表工作

- [[WLA]]: WLA-0 以 Fast-WAM（6B）为主要基线，2B 激活参数即超越其性能
- [[HiMem-WAM]]: 在 LIBERO 上以 97.7% vs 97.6% 略优于 Fast-WAM，并在记忆任务上大幅超越

## 相关概念

- [[扩散世界模型]]
- [[Vision-Language-Action Model]]
- [[自回归策略]]
