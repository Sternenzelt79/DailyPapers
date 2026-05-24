---
type: concept
aliases: [PhysX-Omni, Simulation-Ready 3D Generation]
---

# PhysX-Omni

## 定义
统一的仿真就绪物理 3D 资产生成框架，支持刚体、柔体、铰接体三类物体，直接输出含物理属性（mass, inertia, joint params）的 URDF，可直接用于 MuJoCo 等仿真器。

## 核心要点
1. VLM 分类物体类型（刚/柔/铰接）
2. 针对三类物体分别用专门生成 head（基于 TRELLIS）
3. 输出 URDF + 完整物理属性，sim-ready
4. 前序工作：PhysXVerse（刚体）+ MonoArt（铰接体）

## 代表工作
- [[PhysX-Omni]]: Cao Ziang et al., 2026 (arXiv 2605.21572)

## 相关概念
- [[MuJoCo]]
- [[Isaac Lab]]
