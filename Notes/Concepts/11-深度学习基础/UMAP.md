---
type: concept
aliases: [Uniform Manifold Approximation and Projection, 流形降维, 维度降维]
---

# UMAP（Uniform Manifold Approximation and Projection）

## 定义

UMAP 是一种基于黎曼几何和代数拓扑的非线性降维算法，能将高维特征空间投影到 2D/3D 可视化空间，同时保留局部和全局的数据流形结构，比 t-SNE 更快且可扩展性更好。

## 数学形式

UMAP 最小化高维模糊集与低维模糊集之间的交叉熵：

$$
\mathcal{L} = \sum_{e \in E} w_h(e) \log \frac{w_h(e)}{w_l(e)} + (1 - w_h(e)) \log \frac{1 - w_h(e)}{1 - w_l(e)}
$$

其中 $w_h$、$w_l$ 分别为高维和低维空间中边的权重（模糊集隶属度）。

## 核心要点

1. **流形假设**：假设数据分布在高维空间的低维流形上
2. **保结构降维**：同时保留局部邻域结构（类内聚集）和全局拓扑（类间分离）
3. **速度优势**：比 t-SNE 快 5-10 倍，可处理数百万级别样本
4. **可视化用途**：常用于直观展示特征对齐效果、聚类结构、跨域表征分布

## 代表工作

- [[HARP-VLA]]: 使用 UMAP 可视化对齐前后人类和机器人特征的分布，直观验证 SRPD 损失效果

## 相关概念

- [[余弦距离]]
- [[跨具身学习]]
- [[潜在动作模型]]
