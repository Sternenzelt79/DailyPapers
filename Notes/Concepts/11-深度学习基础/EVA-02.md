---
type: concept
aliases: [EVA-02, EVA CLIP]
---

# EVA-02

## 定义
BAAI 提出的高性能视觉 Transformer 编码器，以 EVA-CLIP 为基础进行改进，在视觉理解任务上达到当时 SOTA。

## 核心要点
1. 基于 [[CLIP]] 预训练，采用 masked image modeling（EVA 目标：预测 CLIP 视觉特征）
2. 在 [[Eagle]]、InternVL 等 MLLM 中广泛用作 vision backbone
3. EVA-02 是第二代，引入 SwiGLU、RoPE 等 Transformer 改进
4. 参数规模从 300M 到 18B，覆盖多个量级

## 代表工作
- Sun et al. (2023), "EVA-02: A Visual Representation Powerhouse"

## 相关概念
- [[CLIP]]
- [[ConvNeXt]]
- [[InternVL]]
