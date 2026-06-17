---
type: concept
aliases: [SE3 alignment, SE(3) alignment, 刚体对齐, Rigid Body Alignment]
---

# SE(3) 对齐

## 定义

SE(3) 对齐是在三维欧氏群 $SE(3)$（旋转+平移）约束下，将两组三维点云对齐的最优化过程，常用于将局部重建坐标系对齐到全局参考坐标系。

## 数学形式

$$
\mathbf{A}^{*} = \arg\min_{\mathbf{A} \in SE(3)} \sum_{t \in \mathcal{S}} \left\| \mathbf{A}\, \mathbf{E}_{t}^{\text{local}} - \mathbf{E}_{t}^{\text{global}} \right\|^{2}
$$

可用 SVD 闭式求解（Umeyama 算法）或 Procrustes 分析。

## 核心要点

1. **保形变换**: $SE(3)$ 仅允许旋转和平移，保持点间距离不变
2. **闭式解**: 当点对应关系已知时，SVD 分解可高效求解
3. **应用场景**: 多视角重建对齐、SLAM 回环检测、视频块坐标系统一

## 代表工作

- [[mu0]]: TraceExtract 中将各视频块的局部重建坐标用 SE(3) 对齐到全局稀疏参考帧，构建全局一致的 3D 轨迹

## 相关概念

- [[Optical Flow]]
- [[VGGT]]
