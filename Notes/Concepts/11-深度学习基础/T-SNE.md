---
type: concept
aliases: [T-SNE, t-SNE, t-Distributed Stochastic Neighbor Embedding]
---

# T-SNE

## 定义
t-分布随机邻域嵌入（t-Distributed Stochastic Neighbor Embedding）：一种非线性降维可视化算法，将高维特征嵌入到 2D/3D 空间，保持局部邻域结构，常用于神经网络特征聚类可视化。

## 数学形式

$$
p_{ij} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}
$$

高维空间的相似度用高斯核，低维空间的相似度 $q_{ij}$ 用 t 分布，最小化 KL 散度。

## 核心要点
1. **局部结构保持**: 高维中相近的点在低维投影中也聚集，适合可视化特征聚类
2. **非凸优化**: 结果受随机初始化影响，多次运行结果可能不同（可用固定 seed 复现）
3. **X-DiffVLA 用途**: 可视化三种机器人具身在关节位置空间的分布聚类，验证 EBF 的判别效果

## 代表工作
- van der Maaten & Hinton, 2008，t-SNE 原始论文
- [[X-DiffVLA]]: Figure 3 用 t-SNE 可视化跨具身动作分布

## 相关概念
- [[Diffusion Model]]
