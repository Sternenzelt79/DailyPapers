---
type: concept
aliases: [Geo-RoPE, Geometric Rotary Position Embedding, 几何旋转位置编码]
---

# Geometric RoPE

## 定义

Geometric RoPE（Geo-RoPE）是对标准 [[RoPE|Rotary Position Encoding]] 的扩展，将相机几何信息（射线方向和相机姿态）编码进注意力机制的位置编码中，使模型能够感知 3D 空间几何结构。

## 数学形式

**Ray 子空间**（像素级，捕获空间变化）：

$$
d^v(h, w) = \text{normalize}\!\left((R^v)^\top (K^v)^{-1} \begin{bmatrix} h + 0.5 \\ w + 0.5 \\ 1 \end{bmatrix}\right)
$$

**Pose 子空间**（视图级，捕获视角关系）：

$$
e^v = [\text{yaw}, \text{pitch}, \text{roll}, t^v, -(R^v)^\top t^v, (R^v)^\top e_z]
$$

**Split-RoPE 编码**（防止干扰）：

$$
\tilde{q} = [\text{RoPE}(q_{\text{ray}}, d^v(h,w));\, \text{RoPE}(q_{\text{pose}}, e^v)]
$$

## 核心要点

1. **双分量设计**: Ray 分量（像素级）+ Pose 分量（视图级），分别编码不同粒度的几何信息
2. **Split 防干扰**: 按头维度拆分为两个子空间，避免空间变化信号与视图级均匀信号相互干扰
3. **3D 坐标系共享**: 所有视角使用同一世界坐标系，使跨视图注意力中的几何约束可传播
4. **无额外参数**: 位置编码本身无可学习参数，几何信息通过旋转矩阵直接注入

## 代表工作

- [[PAIWorld]]: 提出 Geo-RoPE 用于机器人操作多视图世界模型的几何感知建模

## 相关概念

- [[RoPE]]
- [[Cross-View Attention]]
- [[Geometry-Aware Cross-View Attention]]
- [[Split-RoPE]]
