---
type: concept
aliases: [CEM, 交叉熵方法]
---

# Cross-Entropy Method (CEM)

## 定义
基于采样的随机优化 / 黑盒规划方法：迭代用高斯（或其他参数族）采样动作序列，挑出 elite 子集，重新拟合分布。

## 核心要点
1. 不需要梯度，适合不可微动力学。
2. 在世界模型规划里常用：每步采 $N$ 条动作序列、迭代 $K$ 次。
3. 经典配套 [[Model Predictive Control|MPC]]：执行前几步后重规划。

## 代表工作
- [[PETS]]：早期 model-based RL + CEM
- [[LeWM]]：CEM + MPC 在 latent 世界模型上规划（300 samples × 30 iters）

## 相关概念
- [[Model Predictive Control]]
- [[World Model]]
