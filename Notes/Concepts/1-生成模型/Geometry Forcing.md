---
type: concept
aliases: [Geometry Forcing, 几何强制, VideoREPA]
---

# Geometry Forcing

## 定义
一种将视频扩散模型与 3D 几何表示相结合的表示对齐方法，通过特征对齐将几何先验知识注入生成模型，实现一致的世界建模。

## 核心要点
1. 基于表示对齐（Representation Alignment，REPA）范式，在特征层面而非输出层面施加几何约束
2. 利用几何基础模型的中间表示作为监督信号，引导视频生成骨干编码几何一致的结构
3. 与 GEM-4D 类似但几何引导能力相对有限，未专门针对 4D 动态场景设计

## 代表工作
- [[GEM-4D]]: 基于类似原理但采用 4D 几何基础模型（PAGE-4D）蒸馏，在动态机器人操纵场景中表现优于 Geometry Forcing

## 相关概念
- [[Flow Matching]]
- [[Diffusion Transformer]]
- [[视频世界模型]]
- [[几何基础模型]]
