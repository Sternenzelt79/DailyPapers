---
type: concept
aliases: [正则表示, regular representation, 正则特征空间]
---

# Regular Representation

## 定义

正则表示（Regular Representation）是群 $G$ 的一种线性表示，其维度等于群的阶 $|G|$，基向量与群元素一一对应。群元素通过置换基向量（左乘）来作用，是构建等变神经网络中最常用的表示之一。

## 数学形式

对群元素 $g \in G$，正则表示 $\rho_{reg}(g)$ 作用在向量 $v \in \mathbb{R}^{|G|}$ 上：

$$
(\rho_{reg}(g) \cdot v)_h = v_{g^{-1}h} \quad \forall h \in G
$$

即 $g$ 将第 $h$ 个基向量映射到第 $gh$ 个基向量。

正则表示可分解为所有不可约表示的直和（各出现其维度次）：

$$
\rho_{reg} \cong \bigoplus_{\rho \text{ irr.}} \dim(\rho) \cdot \rho
$$

## 核心要点

1. **等变层实现**: 在正则特征空间（regular feature space）中，线性层替换为可操控层（steerable layers），天然满足等变性
2. **通用性**: 正则表示包含了群的所有不可约表示，等变网络可从中自由学习权重分配
3. **计算效率**: 相比完整的 SO(2) 连续群，离散群（如 $C_8$）的正则表示维度低、计算高效
4. **动作分解**: 将机器人动作向量分解为不可约表示的直和，用于等变动作头的设计

## 代表工作

- [[EquiVLA]]: EquiActor 中所有可操控层在正则特征空间中实现，保证 SO(2) 等变动作输出

## 相关概念

- [[SO(2) Equivariance]]
- [[Frame Averaging]]
- [[Steerable CNN]]
