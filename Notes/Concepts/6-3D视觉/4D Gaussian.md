---
type: concept
aliases: [4DGS, 4D Gaussian Splatting, 动态3D高斯]
---

# 4D Gaussian

## 定义
4D Gaussian Splatting（4DGS）是 3D Gaussian Splatting 的时间扩展，用四维高斯基元（3D 空间 + 1D 时间）表示动态场景，支持任意时刻的高质量渲染和一致的时空点追踪。

## 数学形式

每个 4D 高斯基元参数化为：

$$\mathcal{G}(\mathbf{x}, t) = \exp\left(-\frac{1}{2}[\mathbf{x}^\top, t]\ \Sigma_{4D}^{-1}\ [\mathbf{x}^\top, t]^\top\right)$$

- $\Sigma_{4D} \in \mathbb{R}^{4\times4}$：时空协方差矩阵
- 可分解为空间分量 $\Sigma_{3D}$ 和时间运动场 $v(t)$

## 核心要点
1. 时间一致的点追踪：同一物理点在不同帧的 Gaussian 基元保持关联
2. 比 NeRF 渲染速度快数量级，支持实时动态场景重建
3. 在视频世界模型中引入 4D 约束可解决帧间点漂移问题（见 [[GEM-4D]]）
4. 与逆动力学系统结合可从预测视频中提取机器人轨迹

## 代表工作
- [[GEM-4D]]：几何增强视频世界模型，引入 4D 几何一致性约束

## 相关概念
- [[GEM-4D]]
- [[LeWM]]
