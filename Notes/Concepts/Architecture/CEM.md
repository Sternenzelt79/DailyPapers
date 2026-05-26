---
type: concept
aliases: [Cross-Entropy Method, 交叉熵方法, CEM]
---

# CEM（Cross-Entropy Method）

## 定义
一种基于采样的黑盒优化算法，通过迭代更新精英样本的分布参数（均值、方差）来最大化目标函数，常用于模型预测控制中的动作序列优化。

## 数学形式

$$
\begin{aligned}
&\text{1. 采样: } \mathbf{a}^{(i)} \sim \mathcal{N}(\mu, \Sigma), \quad i=1,\ldots,N \\
&\text{2. 评分: } J^{(i)} = \sum_t c(\hat{s}_t^{(i)}, a_t^{(i)}) \\
&\text{3. 精英选取: 取前 } K \text{ 个最优样本} \\
&\text{4. 更新: } \mu \leftarrow \frac{1}{K}\sum_{i \in \text{elite}} \mathbf{a}^{(i)}
\end{aligned}
$$

## 核心要点
1. 无需梯度，适用于不可微代价函数
2. 通过精英样本逐步收窄搜索分布
3. 对高维动作空间效率较低
4. 是世界模型规划中最常用的基础求解器之一

## 代表工作
- [[stable-worldmodel]]: 在 Push-T 操作任务中作为规划求解器使用（2026）

## 相关概念
- [[MPPI]]
- [[模型预测控制]]
- [[世界模型]]
