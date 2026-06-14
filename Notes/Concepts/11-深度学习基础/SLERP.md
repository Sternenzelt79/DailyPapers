---
type: concept
aliases: [SLERP, Spherical Linear Interpolation, 球面线性插值]
---

# SLERP

## 定义
球面线性插值（Spherical Linear Interpolation），一种在旋转空间（SO(3) 或四元数空间）上进行插值的方法，保持插值路径在球面上均匀分布，避免万向节死锁。

## 数学形式

给定两个四元数 $q_1, q_2$ 和插值参数 $t \in [0, 1]$：

$$
\text{SLERP}(q_1, q_2; t) = q_1 \left( q_1^{-1} q_2 \right)^t
$$

等价形式（基于夹角 $\Omega = \arccos(q_1 \cdot q_2)$）：

$$
\text{SLERP}(q_1, q_2; t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} q_1 + \frac{\sin(t\Omega)}{\sin\Omega} q_2
$$

## 核心要点
1. 在旋转流形上插值，路径长度均匀，角速度恒定
2. 优于线性插值（LERP）：LERP 在旋转空间中会产生不均匀速度和非最短路径
3. 广泛用于动画、机器人轨迹平滑、姿态估计后处理

## 代表工作
- [[GEM-4D]]: 在 Adaptive Inverse Dynamic System 中，当 FoundationPose 置信度低时，用 SLERP 在时序上插值恢复丢失帧的旋转估计

## 相关概念
- [[FoundationPose]]
- [[帧间对应]]
- [[7-规划与控制]]
