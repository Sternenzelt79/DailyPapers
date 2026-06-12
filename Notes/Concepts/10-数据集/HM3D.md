---
type: concept
aliases: [HM3D, Habitat-Matterport 3D, HM3D Dataset]
---

# HM3D

## 定义
Habitat-Matterport 3D Dataset，一个大规模室内 3D 扫描数据集，包含真实建筑的高精度三维重建，广泛用于具身 AI 仿真中的视觉导航任务训练。

## 核心要点
1. 包含 1,000+ 个真实建筑场景（NavWAM 使用其中 802 个）
2. 与 Habitat 仿真器深度集成，支持高效的自我中心视角渲染
3. NavWAM Phase 1 预训练生成了 185,000 条导航轨迹

## 代表工作
- [[NavWAM]]: 使用 HM3D 的 802 个场景生成 185,000 条仿真轨迹，用于 Phase 1 预训练

## 相关概念
- [[GNM]]
- [[ViNT]]
