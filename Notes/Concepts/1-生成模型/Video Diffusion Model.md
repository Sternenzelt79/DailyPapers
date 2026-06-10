---
type: concept
aliases: [视频扩散模型, Video Diffusion, VDM]
---

# Video Diffusion Model

## 定义
将扩散模型框架扩展到视频时序数据，通过联合时序帧的去噪过程生成高质量视频。

## 核心要点
1. 在潜空间中对时序帧的联合分布建模，保持时序一致性
2. 通常使用 3D 卷积或时序注意力机制捕获帧间依赖
3. 代表架构包括基于 DiT（Diffusion Transformer）的 WAN、Wan2.2 等

## 代表工作
- [[Efficient-WAM]]: 从 WAN-2.2-5B 蒸馏紧凑视频扩散专家用于机器人控制

## 相关概念
- [[Diffusion Model]]
- [[Flow Matching]]
- [[Video Prediction]]
- [[World-Action Model]]
