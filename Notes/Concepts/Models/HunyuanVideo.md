---
type: concept
aliases: [HunyuanVideo, Hunyuan Video]
---

# HunyuanVideo

## 定义
HunyuanVideo 是腾讯开源的大规模视频生成基础模型，基于双向扩散 Transformer（DiT）架构，支持高质量长视频生成，被广泛用作视频世界模型的基础模型。

## 核心要点
1. 双向注意力机制支持全局时序一致性，但天然不支持因果自回归生成
2. 开源权重使其成为 BiWM 等视频世界模型研究的起点
3. 与 LTX-2、Wan2.1 等并列为当前主流开源视频生成基础模型

## 代表工作
- [[BiWM]]: 以 HunyuanVideo 为基底，通过 PackForcing 技术转换为自回归交互式世界模型

## 相关概念
- [[视频世界模型]]
- [[扩散模型]]
- [[Wan2.1]]
