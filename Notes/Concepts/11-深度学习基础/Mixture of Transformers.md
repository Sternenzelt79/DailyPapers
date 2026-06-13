---
type: concept
aliases: [MoT, Mixture of Transformers]
---

# Mixture of Transformers

## 定义
一种多模态 Transformer 架构，通过专用的模态特定专家（expert）模块并行处理不同模态（如 RGB 视频、Mask、动作序列），同时共享通用注意力机制以实现跨模态交互。

## 核心要点
1. **模态专用处理**: 每种模态（视频/掩码/动作）由专用 expert 处理，避免不同模态的表示相互干扰
2. **共享注意力**: 跨模态交互通过共享 self-attention 层实现，使各模态的 token 能相互 attend
3. **统一训练**: 与 [[Mixture of Experts]]（MoE）不同，MoT 按模态路由而非按 token 路由，适合多任务多模态联合生成

## 代表工作
- [[MaskWAM]]: 使用 MoT 统一处理 RGB 潜变量、Mask 潜变量和动作序列的联合去噪，实现三路 [[Flow Matching]] 训练

## 相关概念
- [[Diffusion Transformer]]
- [[Flow Matching]]
- [[World Action Model]]
- [[Video VAE]]
