---
type: concept
aliases: [Centered Kernel Alignment, 中心核对齐]
---

# CKA

## 定义
Centered Kernel Alignment，一种衡量两个表征空间（如神经网络不同层的激活）之间相似度的指标，对旋转和各向同性缩放不变。

## 数学形式
$$\text{CKA}(K, L) = \frac{\|L^\top K\|_F^2}{\|K^\top K\|_F \cdot \|L^\top L\|_F}$$

其中 $K, L$ 为中心化后的 Gram 矩阵。

## 核心要点
1. 对正交变换不变，因此可以跨架构比较表征
2. 用于识别神经网络层间的冗余性（如 VLA 层剪枝）
3. 线性 CKA 和 RBF kernel CKA 是最常用的变体

## 代表工作
- [[CLP]]: 用 CKA 识别 VLA 中的冗余 twin layers，实现训练无关压缩

## 相关概念
- [[Transformer]]
- [[VLA]]
