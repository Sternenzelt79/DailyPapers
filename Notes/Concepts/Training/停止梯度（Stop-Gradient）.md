---
type: concept
aliases: [Stop-Gradient, sg, detach, stop_gradient]
---

# 停止梯度（Stop-Gradient）

## 定义

停止梯度（Stop-Gradient，记为 $\text{sg}[\cdot]$ 或 `detach`）是一种训练技巧：在前向计算中保持张量值不变，但在反向传播时**阻断梯度流**，使目标侧不参与梯度更新，仅通过预测侧优化参数。

## 数学形式

$$
\text{sg}[x] = x, \quad \frac{\partial \,\text{sg}[x]}{\partial x} = 0
$$

即值不变，梯度为零。

## 核心要点

1. **防止表示坍塌**：在自监督学习（如 BYOL、SimSiam、NextLat）中，若目标和预测侧均参与优化，模型会退化为输出常数（坍塌解）；stop-gradient 打破对称性
2. **知识蒸馏**：冻结教师网络参数等效于对教师侧施加 stop-gradient
3. **Bootstrap 目标**：EMA 目标网络 + stop-gradient 是稳定自监督训练的标准做法
4. **实现**：PyTorch 中为 `tensor.detach()`，JAX/TF 中为 `stop_gradient()`

## 代表工作

- [[NextLat]]: 对目标隐状态 $h_{t+i}$ 施加 stop-gradient，防止动力学网络学到平凡解
- [[BYOL]]: 动量编码器 + stop-gradient 实现无负样本对比学习

## 相关概念

- [[表示坍塌]]
- [[知识蒸馏]]
- [[EMA]]
