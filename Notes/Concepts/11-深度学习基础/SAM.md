---
type: concept
aliases: [Segment Anything Model, 分割任意模型]
---

# SAM

## 定义
Meta AI 提出的通用图像分割基础模型，支持通过点、框、文本等多种 prompt 对任意对象进行零样本分割。

## 核心要点
1. 三组件架构：Image Encoder（MAE 预训练 ViT）+ Prompt Encoder + Mask Decoder
2. 训练数据 SA-1B：超过 10 亿张 mask，规模远超已有分割数据集
3. 支持 prompt 类型：点、边界框、文本（有限）、mask
4. 后续版本 [[SAM2]] 扩展到视频分割

## 代表工作
- [[SAM]]: Kirillov et al. (2023)，Meta AI

## 相关概念
- [[SAM2]]
- [[GroundingDINO]]
- [[FoundationPose]]
