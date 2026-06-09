---
type: concept
aliases: [CityDreamer]
---

# CityDreamer

## 定义
CityDreamer 是一种无边界 3D 城市生成模型，将城市建筑和自然景观分开建模，支持自由视角渲染和大规模城市场景生成。

## 核心要点
1. 分离式生成：建筑物用实例感知 NeRF，天空/地面用背景 NeRF
2. OSM 地图作为条件，控制城市布局
3. 支持无边界城市飞行视角渲染

## 代表工作
- Xie et al. (2024), CityDreamer: Compositional Generative Model of Unbounded 3D Cities

## 相关概念
- [[DriveDreamer]]
- [[SceneDreamer]]
- [[NeRF]]
