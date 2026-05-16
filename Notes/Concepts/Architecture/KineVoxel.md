---
type: concept
aliases: [Kinematic Voxel, KineVoxel Injection]
---

# KineVoxel

## 定义
PhysForge 提出的一种将运动学约束（关节轴、枢轴）注入到 3D 扩散生成过程的表示方法，通过体素化的运动学参数引导物理感知的 3D 资产生成。

## 数学形式
$$
\mathbf{K} = \{(\mathbf{v}_i, \mathbf{a}_i, \mathbf{p}_i)\}_{i=1}^{N_{\text{joint}}}
$$

其中 $\mathbf{v}_i$ 为体素位置，$\mathbf{a}_i$ 为关节轴方向，$\mathbf{p}_i$ 为枢轴点位置。将 $\mathbf{K}$ 作为条件注入扩散模型的 cross-attention 层。

## 核心要点
1. 将关节运动学参数（轴、枢轴）体素化，转化为可与 3D 扩散模型兼容的表示
2. 通过 KVI（KineVoxel Injection）机制将物理约束融入生成过程
3. 允许生成不仅外观合理、而且运动学正确的可交互 3D 资产
4. 评估指标 Joint-Axis-Err 和 Joint-Pivot-Err 专为此类任务设计

## 代表工作
- [[PhysForge]]: 提出 KineVoxel 用于物理感知 3D 资产生成

## 相关概念
- [[3D Gaussian Splatting]]
- [[PhysForge]]
