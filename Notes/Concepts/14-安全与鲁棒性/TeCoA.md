---
type: concept
aliases: [TeCoA, Text-guided Contrastive Adversarial training]
---

# TeCoA

## 定义
一种针对多模态大模型（MLLM/CLIP）的对抗鲁棒性增强方法，通过文本引导的对比对抗训练提升视觉特征对对抗扰动的鲁棒性。

## 核心要点
1. 在图像编码器训练中引入对抗样本
2. 利用文本监督信号对齐干净图像和对抗图像的特征空间
3. 在 CLIP-based 模型上效果显著

## 代表工作
- [[Robust-U1]]: 作为对抗鲁棒性 baseline 进行对比

## 相关概念
- [[分布外泛化]]
- [[对抗性扰动]]
- [[CLIP]]
