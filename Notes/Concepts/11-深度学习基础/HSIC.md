---
type: concept
aliases: [Hilbert-Schmidt Independence Criterion, 希尔伯特-施密特独立性准则]
---

# HSIC

## 定义
HSIC（Hilbert-Schmidt Independence Criterion）是一种基于再生核希尔伯特空间（RKHS）的统计独立性度量，用于衡量两个随机变量之间的依赖强度。HSIC = 0 当且仅当两变量统计独立。

## 数学形式

$$\text{HSIC}(X, Y) = \frac{1}{(n-1)^2} \text{tr}(KHLH)$$

- $K, L$：分别为 $X, Y$ 的核矩阵（如 RBF 核）
- $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$：中心化矩阵
- $\text{tr}(\cdot)$：矩阵迹

## 核心要点
1. 无参数，不假设线性关系，可捕捉任意非线性依赖
2. 核函数选择影响灵敏度：RBF 核对连续变量效果好
3. 在深度学习中常用作：特征解耦损失、重要性度量、层间依赖分析
4. 在 VLA 量化中（ActQuant）用于识别对动作输出最关键的权重张量

## 代表工作
- [[ActQuant]]：用 HSIC 指导 VLA sub-4-bit 量化的跨张量位宽分配

## 相关概念
- [[LoRA]]
