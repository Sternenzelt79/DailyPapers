---
type: concept
aliases: [倒角距离, CD, Chamfer Distance]
---

# Chamfer Distance（倒角距离）

## 定义

衡量两个点云之间相似度的非对称或对称距离度量，通过计算每个点到另一点云最近邻点的距离之和来量化两点集的差异。

## 数学形式

$$
\text{CD}(P, Q) = \frac{1}{|P|} \sum_{p \in P} \min_{q \in Q} \|p - q\|^2 + \frac{1}{|Q|} \sum_{q \in Q} \min_{p \in P} \|q - p\|^2
$$

## 核心要点

1. 对点云大小不敏感，可比较不同数量点的点集
2. 计算复杂度为 $O(|P| \cdot |Q|)$，可用 KD-tree 加速
3. 在 3D 重建、点云生成中广泛用作损失函数
4. 变体 $\text{CD}_z$ 仅计算 z 轴方向距离，用于接触平面约束

## 代表工作

- [[GRAIL]]: 用于深度对齐损失 $\mathcal{L}_{depth}$ 和接触损失 $\mathcal{L}_{cont}$，约束重建点云与深度估计点云一致

## 相关概念

- [[FoundationPose]]
- [[HOI]]
