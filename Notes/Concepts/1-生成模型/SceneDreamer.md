---
type: concept
aliases: [SceneDreamer, 场景生成模型]
---

# SceneDreamer

## 定义
SceneDreamer 是一个从随机噪声无条件生成无边界 3D 自然场景的生成模型，利用 BEV（鸟瞰图）语义地图作为场景布局先验。

## 核心要点
1. 从 BEV 语义地图生成高度图和语义特征
2. 用 NeRF 风格渲染 360° 无边界场景
3. 分层生成：天空、远景、近景分别处理

## 代表工作
- Chen et al. (2023), SceneDreamer: Unbounded 3D Scene Generation from 2D Image Collections

## 相关概念
- [[CityDreamer]]
- [[DriveDreamer]]
- [[NeRF]]
