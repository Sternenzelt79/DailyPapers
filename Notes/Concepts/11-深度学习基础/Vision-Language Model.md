---
type: concept
aliases: [VLM, 视觉语言模型]
---

# Vision-Language Model

## 定义

同时处理视觉（图像/视频）和语言（文本）输入的多模态大模型，通过大规模预训练学习视觉-语言对齐，支持图文问答、图像描述等任务。

## 核心要点

1. **多模态对齐**：通过对比学习或自回归目标对齐视觉和语言特征空间
2. **作为 VLA Backbone**：VLM 常被用作 VLA 的 Backbone，提供强大的语义理解和泛化能力
3. **代表性模型**：Qwen3-VL, LLaVA, InternVL, PaliGemma 等

## 相关概念

- [[Vision-Language-Action Model]]
- [[Transformer]]

## 代表工作

- [[LaWAM]]: 使用 Qwen3-VL 前 16 层（Qwen-GR00T）作为策略 VLM Backbone
