---
type: concept
aliases: [Huber Loss, Huber 损失, 平滑L1损失, Smooth L1 Loss]
---

# Huber 损失

## 定义

Huber 损失是一种结合 L1 和 L2 损失优点的回归损失函数，在残差较小时表现为 L2（平方），在残差较大时表现为 L1（线性），对异常值具有鲁棒性。

## 数学形式

$$
L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2, & |y - \hat{y}| \leq \delta \\
\delta \cdot \left(|y - \hat{y}| - \frac{\delta}{2}\right), & |y - \hat{y}| > \delta
\end{cases}
$$

## 核心要点

1. **鲁棒性**: 相比 L2 损失，对异常值（outlier）不敏感，不会因个别大误差主导梯度
2. **平滑性**: 相比 L1 损失，在 $y = \hat{y}$ 处可导，数值优化更稳定
3. **超参数 $\delta$**: 控制 L1/L2 切换的阈值，常用值为 1.0；$\delta \to 0$ 趋近 L1，$\delta \to \infty$ 趋近 L2

## 代表工作

- [[HABC]]: 效率头 $\hat{V}_e$ 训练时使用 Huber 损失（$\delta=1.0$），提升对异常步数估计的鲁棒性

## 相关概念

- [[二元交叉熵]]
- [[联合Critic损失]]
