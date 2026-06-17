---
type: concept
aliases: [Z-Buffer, 深度缓冲, Depth Buffer, z-buffering]
---

# Z-Buffering

## 定义

Z-Buffering（深度缓冲）是计算机图形学中用于解决遮挡问题的经典算法：为每个像素维护一个"深度值"，只渲染深度最小（即离相机最近）的片元，从而自动处理物体遮挡关系。

## 数学形式

对于目标视角 $t$ 的每个像素格 $(u,v)$，从所有投影到该格的 3D 点集 $\Omega^t(u,v)$ 中选取相机坐标系深度最小的点：

$$
i^*(u,v) = \arg\min_{i \in \Omega^t(u,v)} [\mathbf{E}^t \mathbf{p}_i]_z
$$

## 核心要点

1. **遮挡处理**：确保近处物体遮住远处物体，符合物理规律。
2. **逐像素判断**：每个像素独立维护当前最近点的深度，渲染时逐个更新。
3. **延伸到潜空间**：Mirage 将 Z-Buffering 从 RGB 像素空间移植到 VAE 潜分辨率网格，直接在潜空间完成可见性判断，避免像素级渲染。

## 代表工作

- [[Mirage]]: 将 Z-Buffering 应用于潜分辨率空间，实现单次投影直接生成目标视角潜特征，无需 RGB 光栅化

## 相关概念

- [[针孔相机模型]]
- [[Point Cloud Warping]]
- [[DepthAnything]]
