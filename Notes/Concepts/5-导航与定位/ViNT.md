---
type: concept
aliases: [ViNT, Visual Navigation Transformer]
---

# ViNT

## 定义
ViNT（Visual Navigation Transformer）是一个基于 Transformer 的通用视觉导航基础模型，在多平台数据上预训练后，能泛化到新机器人平台的图像目标导航任务。

## 核心要点
1. 用 Transformer 建模长时间依赖，处理历史观测序列
2. 在 GNM 数据集（7 种机器人）上预训练
3. 通过 LoRA 等方式快速适配新平台

## 代表工作
- Shah et al. (2023), ViNT: A Foundation Model for Visual Navigation

## 相关概念
- [[GNM]]
- [[NoMaD]]
- [[WAM-Nav]]
