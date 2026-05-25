---
type: concept
aliases: [ManiGaussian]
---

# ManiGaussian

## 定义
基于 3D Gaussian Splatting 的机器人操作方法，将 3DGS 引入机器人策略学习，用高斯场表示场景并从中学习操作策略；GAF 是其时序扩展版本。

## 核心要点
1. 将 3D 场景表示为高斯点集，每个高斯点携带位置、颜色、不透明度等属性
2. 用高斯场的渲染图像替代原始 RGB 输入，提供更一致的场景表示
3. 相较 NeRF 渲染速度快，支持实时策略学习
4. 局限：高斯初始化对场景质量敏感，遮挡场景下高斯重建不稳定

## 代表工作
- [[GAF]]: 在 ManiGaussian 基础上加入时序（4D）高斯动作场，扩展到动态世界建模

## 相关概念
- [[GAF]]
- [[Diffusion Policy]]
- [[扩散世界模型]]
