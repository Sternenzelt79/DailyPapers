---
type: concept
aliases: [WAN-2.2, WAN-5B, Wan Video, 万象视频]
---

# WAN

## 定义
阿里巴巴通义万象团队发布的大规模视频生成基础模型，WAN-2.2 系列包含 5B 参数版本，在视频生成质量上达到 SOTA 水平。

## 核心要点
1. 基于 Diffusion Transformer 架构，使用流匹配训练目标
2. WAN-2.2-5B 被 Efficient-WAM 等工作用作视频生成教师模型
3. 通过层切片可压缩为 ~0.8B 参数的紧凑版本，同时通过蒸馏保持运动建模能力

## 代表工作
- [[Efficient-WAM]]: 从 WAN-2.2-5B 蒸馏 1B 紧凑视频专家

## 相关概念
- [[Video Diffusion Model]]
- [[Flow Matching]]
- [[Knowledge Distillation]]
