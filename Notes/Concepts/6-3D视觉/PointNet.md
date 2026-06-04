---
type: concept
aliases: [PointNet, 点云深度学习]
---

# PointNet

## 定义
PointNet 是第一个直接在原始点云上操作的深度学习网络，用对称函数（max pooling）保证点集的置换不变性，实现 3D 物体分类和分割。

## 数学形式
$$f(\{x_1,...,x_n\}) = g(h(x_1), ..., h(x_n))$$

其中 $g$ 是对称函数（max pooling），$h$ 是逐点 MLP。

## 核心要点
1. 置换不变性：通过 max pooling 实现
2. 无需体素化，直接处理原始点云
3. PointNet++ 引入局部邻域聚合，提升层次特征学习

## 代表工作
- Qi et al. (2017), PointNet; Qi et al. (2017), PointNet++

## 相关概念
- [[PointAction]]
- [[3D Gaussian Splatting]]
