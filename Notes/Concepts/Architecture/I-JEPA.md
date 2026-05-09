---
type: concept
aliases: [I-JEPA, Image-JEPA]
---

# I-JEPA

## 定义
图像版 [[JEPA]]：在像素 patch 上做"上下文 → 目标 patch"的 latent 预测，使用 EMA target encoder。Assran et al., CVPR 2023。

## 核心要点
1. 不重建像素，只在 latent 上做预测，避免浪费容量在低层细节。
2. 用 multi-block masking + EMA target，下游 linear probe 接近 DINO。
3. LeCun 阵营 JEPA 系列在图像上的首个稳定实例。

## 代表工作
- I-JEPA (Assran 2023)
- [[V-JEPA]] / V-JEPA 2：视频版
- [[LeWM]]：把 JEPA 思想推到端到端世界模型

## 相关概念
- [[JEPA]]
- [[V-JEPA]]
- [[Vision Transformer]]
