---
type: concept
aliases: [ConvNeXt, ConvNet 2020s]
---

# ConvNeXt

## 定义
Meta AI 提出的现代化纯卷积网络，借鉴 Vision Transformer 的设计理念对 ResNet 进行系统性改进，证明卷积网络仍能与 ViT 竞争。

## 核心要点
1. 从 ResNet 出发，逐步吸收 Swin Transformer 的训练策略（AdamW、Mixup、大 kernel）
2. 关键改进：7×7 深度可分离卷积、倒瓶颈结构、GELU 激活、LayerNorm 替代 BN
3. 在 ImageNet 分类、目标检测等任务与同等参数 ViT 相当
4. 在 [[Eagle]] 等 MLLM 中用作 vision expert 编码器之一

## 代表工作
- Liu et al. (2022), "A ConvNet for the 2020s"

## 相关概念
- [[CLIP]]
- [[EVA-02]]
- [[Vision Transformer]]
