---
type: concept
aliases: [MagicBrush, 指令引导图像编辑数据集]
---

# MagicBrush

## 定义
MagicBrush 是一个人工标注的指令引导图像编辑数据集和基准，包含多轮编辑数据，用于训练和评估图像编辑模型。

## 核心要点
1. 人工在 DALL-E 生成图上标注编辑指令和结果
2. 支持多轮连续编辑（track 格式）
3. 常用于评估 VLM 的 forward dynamics prediction 能力

## 代表工作
- Zhang et al. (2024), MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing

## 相关概念
- [[SmartEdit]]
- [[VLM-FDP]]
