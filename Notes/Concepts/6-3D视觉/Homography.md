---
type: concept
aliases: [单应矩阵, 单应变换, Homography Matrix, Projective Transformation]
---

# Homography（单应矩阵）

## 定义

单应矩阵是描述同一场景在两个不同平面（或两个相机视角）之间的投影变换关系的 $3 \times 3$ 矩阵，在齐次坐标下通过矩阵乘法实现像素坐标的互相映射。

## 数学形式

$$
(u,\, v,\, 1)^T \sim H\, (u',\, v',\, 1)^T
$$

其中 $H$ 为 $3 \times 3$ 的单应矩阵，$\sim$ 表示在齐次坐标意义下的相等（允许任意尺度因子）。

展开形式：

$$
\begin{pmatrix} u \\ v \\ 1 \end{pmatrix} \sim \begin{pmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{pmatrix} \begin{pmatrix} u' \\ v' \\ 1 \end{pmatrix}
$$

## 核心要点

1. **自由度**: 单应矩阵有 8 个自由度（9 个参数减去 1 个尺度），因此至少需要 4 对点对应关系来求解。
2. **应用场景**: 图像拼接（全景图）、图像校正、多传感器空间对齐、增强现实等。
3. **估计方法**: DLT（直接线性变换）算法，RANSAC 鲁棒估计。
4. **单应矩阵 vs 基础矩阵**: 单应矩阵适用于平面场景或纯旋转相机；基础矩阵适用于一般三维场景。

## 代表工作

- [[MuseVLA]]: 用单应矩阵将热成像、声学、mmWave 传感器坐标对齐到 RGB 图像坐标系，实现一次性离线标定后的多传感器空间融合。

## 相关概念

- [[6-3D视觉/Homography|Homography]]
- [[SAM-3]]
- [[mmWave 雷达]]
