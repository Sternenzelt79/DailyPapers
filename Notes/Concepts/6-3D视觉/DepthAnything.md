---
type: concept
aliases: [Depth Anything, 深度估计基础模型]
---

# DepthAnything

## 定义
DepthAnything 是一个基于大规模数据训练的单目深度估计基础模型，在多样场景下实现强泛化的相对深度估计。

## 核心要点
1. 用 66M 标注图像 + 62M 未标注图像联合训练
2. 基于 DINOv2 视觉编码器，强视觉先验
3. 输出相对深度，可通过 scale + shift 对齐到 metric depth

## 代表工作
- Yang et al. (2024), Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data
- Depth Anything V3 (2025): 多视图深度估计，用于 GEM-4D 在 Droid 数据集（无 GT 深度）上的深度估计
- [[GEM-4D]]: 使用 Depth Anything V3 作为真实域评估的深度估计工具，同时也是几何基础模型的选项之一

## 相关概念
- [[DINOv2]]
- [[6-3D视觉]]
