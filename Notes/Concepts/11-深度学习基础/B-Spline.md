---
type: concept
aliases: [B样条, B-spline, 三次样条, Cubic Spline]
---

# B-Spline

## 定义

B-Spline（基样条）是一种用有限个控制点参数化平滑曲线的数学工具，通过基函数线性组合保证曲线的局部支撑和全局平滑性。

## 数学形式

$$
\mathbf{C}(t) = \sum_{i=0}^{D-1} \mathbf{P}_i \, B_{i,k}(t)
$$

其中 $B_{i,k}(t)$ 为 $k$ 阶 B-spline 基函数，$\mathbf{P}_i$ 为控制点，$D$ 为控制点数量。

矩阵形式：$\mathbf{T} = \mathbf{B} \mathbf{P}$，$\mathbf{B} \in \mathbb{R}^{H \times D}$ 为基矩阵。

## 核心要点

1. **局部控制**: 修改一个控制点只影响局部曲线段，不改变全局形状
2. **平滑性**: 三次 B-spline 具有 $C^2$ 连续性（曲率连续），适合轨迹参数化
3. **压缩表征**: $D$ 个控制点可表示 $H \gg D$ 个轨迹采样点，自然正则化
4. **Tikhonov 正则**: 对差分矩阵 $\boldsymbol{\Gamma}\mathbf{P}$ 施加正则化可进一步控制平滑度

## 代表工作

- [[mu0]]: 用 $D=10$ 个三次 B-spline 控制点参数化 $H=32$ 帧的 3D 交互轨迹，用 Flow Matching 在控制点空间上训练

## 相关概念

- [[Conditional Flow Matching]]
- [[Flow Matching]]
