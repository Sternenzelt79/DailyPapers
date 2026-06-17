---
type: concept
aliases: [PAGE-4D, Pose and Geometry Estimation 4D]
---

# PAGE-4D

## 定义
一种 4D 几何基础模型，通过运动感知掩码（motion-aware masking）解耦静态背景与动态物体，提供鲁棒的位姿和几何估计，专为动态场景设计。

## 核心要点
1. 在 VGGT 等几何 transformer 基础上，通过运动感知掩码分离静态与动态成分，提升动态场景下的几何鲁棒性
2. 中间表示编码了帧间对应所需的全部几何因子：深度 $D$、相机旋转 $\mathbf{R}$、平移 $\mathbf{T}$ 和场景流 $\Delta\mathbf{X}$
3. 在 GEM-4D 消融中优于 VGGT，因其更适合机器人操纵等动态场景

## 数学形式

页间对应关系：

$$
\mathbf{p}_{t+1} \sim \mathbf{K} \left[ \mathbf{R}_{t \rightarrow t+1} \mathbf{D}(\mathbf{p}_t) \mathbf{K}^{-1} \mathbf{p}_t + \mathbf{T}_{t \rightarrow t+1} + \Delta\mathbf{X}_t \right]
$$

## 代表工作
- [[GEM-4D]]: 将 PAGE-4D 冻结用作 correspondence teacher，通过特征蒸馏提升视频 DiT 的几何一致性

## 相关概念
- [[VGGT]]
- [[几何基础模型]]
- [[帧间对应]]
- [[Depth Anything]]
