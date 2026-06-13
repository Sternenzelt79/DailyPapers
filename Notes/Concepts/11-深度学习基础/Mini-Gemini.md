---
type: concept
aliases: [Mini-Gemini, MGM]
---

# Mini-Gemini

## 定义
CUHK 提出的高效多模态大语言模型，通过双视觉编码器（低分辨率语义 + 高分辨率细节）和 Any-resolution 处理，在小模型规模下达到强视觉理解性能。

## 核心要点
1. 使用双 encoder：低分辨率 CLIP 提供语义，高分辨率 ConvNet 提供细节
2. 支持 2B 到 34B 多种规模，均在主流 benchmark 上有竞争力
3. 为 MLLM 设计空间探索（如 Eagle）的重要 baseline

## 代表工作
- Eagle: 以 Mini-Gemini 为对比 baseline

## 相关概念
- [[CLIP]]
- [[ConvNeXt]]
- [[InternVL]]
