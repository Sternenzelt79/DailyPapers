---
type: concept
aliases: [SmolVLM2-2.2B, SmolVLM]
---

# SmolVLM2

## 定义

SmolVLM2 是 HuggingFace 发布的轻量级视觉语言模型系列，以 SigLIP 视觉编码器 + 小型文本解码器为架构，在推理效率和多模态理解之间取得平衡。

## 核心要点

1. **轻量设计**: 2.2B 参数版本在性能和效率之间达到良好平衡，适合作为机器人策略骨干
2. **SigLIP 视觉编码器**: 对 RGB 图像和深度图的 patch 特征提取
3. **可模块化**: 骨干可冻结后通过适配器（如 Gated Cross-Attention）接入下游任务

## 代表工作

- [[mu0]]: 用 SmolVLM2-2.2B 作为语义编码骨干，提取条件特征供 Trace Expert 使用，深度图经独立 patch stem 后与 RGB token 共享 SigLIP 层

## 相关概念

- [[VLM]]
- [[DINOv2]]
- [[Gated Cross-Attention]]
