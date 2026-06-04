---
type: concept
aliases: [NoMaD, No Map Diffusion, 无地图扩散导航]
---

# NoMaD

## 定义
NoMaD（No Map Diffusion）是一个无需预建地图的扩散策略导航模型，将 diffusion policy 用于视觉目标导航，在无结构化环境中实现探索和目标导向行为统一。

## 核心要点
1. 扩散策略统一建模探索和 goal-reaching 两种行为
2. 不需要地图或精确定位，直接从图像观察输出动作
3. 在 GNM 数据集上预训练，跨平台迁移

## 代表工作
- Sridhar et al. (2024), NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration

## 相关概念
- [[GNM]]
- [[ViNT]]
- [[Diffusion Policy]]
