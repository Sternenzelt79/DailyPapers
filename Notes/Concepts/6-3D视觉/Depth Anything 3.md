---
type: concept
aliases: [Depth Anything V3, DepthAnything3, DA3]
---

# Depth Anything 3

## 定义

一种针对复杂场景（包括机器人操作场景）优化的多视角深度估计模型，在 Depth Anything 系列基础上进一步提升了多视角一致性和操作场景的精度。

## 核心要点

1. 支持多视角输入，提升跨视角深度一致性
2. 在操作场景中相比通用深度模型精度更高，但仍存在厘米级误差
3. 输出深度图可直接反投影为 3D 点云，用于构建接触几何关系
4. 被 [[iMaC]] 用于估计场景点云 $P^s_t$，构建 [[Contact Images]]

## 局限性

- 在机器人操作场景中存在厘米级误差，影响接触时机和碰撞定位精度
- 对于 iMaC 等需要精确接触几何的任务，仍有改进空间

## 代表工作

- Lin et al. (2025): Depth Anything 3 原始论文
- [[iMaC]]: 使用 Depth Anything 3 进行多视角深度估计并构建接触图像
- [[GAM]]: 将 DA3-Giant（在 Track4World 上微调）作为几何骨干网络，分割后插入因果预测器实现机器人操控策略

## 相关概念

- [[DepthAnything]]
- [[Contact Images]]
- [[6-3D视觉]]
