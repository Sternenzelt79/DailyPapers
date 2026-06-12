---
type: concept
aliases: [Cosmos Predict2, Cosmos-Predict2]
---

# Cosmos Predict2

## 定义
NVIDIA 开发的 2B 参数视频扩散-Transformer 基础模型，采用因果 VAE 编解码结合 DiT 去噪骨干，支持文本条件的视频生成，被广泛用作机器人策略的预训练骨干。

## 核心要点
1. 采用[[因果 VAE]]将视频帧压缩到空间-时序潜变量，保持时序因果性
2. DiT（[[Diffusion Transformer]]）骨干对潜变量序列执行去噪
3. 支持文本条件接口，便于下游任务的条件化微调
4. 作为视频生成预训练骨干，迁移到机器人策略时提供丰富的视觉先验

## 代表工作
- [[NavWAM]]: 基于 Cosmos Predict2 微调的导航世界行动模型，引入九帧潜在画布联合预测动作、未来观测和目标进度值

## 相关概念
- [[Diffusion Transformer]]
- [[因果 VAE]]
- [[World-Action Model]]
