---
type: concept
aliases: [ABot M0, ActionBot-M0]
---

# ABot-M0

## 定义
基于 [[Qwen3-VL]] 的 VLA 模型，以 [[DiT|DiT]]-based [[Flow Matching|流匹配]] 作为动作生成器，是 [[WorldPilot]] 的基座架构和主要比较基线。

## 核心要点
1. VLM backbone：Qwen3-VL，负责图像和语言的多模态编码
2. 动作生成器：DiT-based 流匹配，生成 [[Action Chunking|动作块]]
3. LIBERO-Plus 零样本 OOD 基准成功率：80.5%（Total）

## 代表工作
- [[WorldPilot]]: 在 ABot-M0 基础上增加 [[Latent Steering]] 和 [[Action Steering]]，将 LIBERO-Plus 成功率从 80.5% 提升至 84.7%

## 相关概念
- [[Qwen3-VL]]
- [[DiT]]
- [[Flow Matching]]
- [[VLA]]
