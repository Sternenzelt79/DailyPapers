---
type: concept
aliases: [Janus, Janus-Pro, DeepSeek Janus]
---

# Janus

## 定义
DeepSeek 提出的统一多模态模型，将视觉理解（分析）和视觉生成解耦为两条处理路径，通过双路编码器避免理解和生成任务之间的表示空间冲突。

## 核心要点
1. 解耦理解编码器（高语义密度）和生成编码器（低级像素信息）
2. 统一 Transformer 解码器负责两种任务
3. 与 [[SenseNova-U1]] 等完全统一架构相比，走"解耦路线"

## 代表工作
- Janus-Pro（DeepSeek）: 在多模态理解和生成 benchmark 上同时取得强结果

## 相关概念
- [[OmniGen2]]
- [[BAGEL]]
- [[SenseNova-U1]]
- [[统一多模态模型]]
