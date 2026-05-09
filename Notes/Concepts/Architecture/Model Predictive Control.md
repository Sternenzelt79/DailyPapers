---
type: concept
aliases: [MPC, 模型预测控制]
---

# Model Predictive Control (MPC)

## 定义
在每个时刻基于动力学模型在有限 horizon $H$ 内求解最优控制序列，仅执行前 $K$ 步后重新规划，循环往复。

## 数学形式
$$
a^*_{1:H} = \arg\min_{a_{1:H}} \sum_{t=1}^{H} \ell(s_t, a_t) + \ell_T(s_H)
$$

## 核心要点
1. 闭环 receding-horizon 控制，对模型误差鲁棒。
2. 内层求解器可为梯度下降、CEM、iLQR 等。
3. 在 [[World Model]] 范式中与 [[Cross-Entropy Method|CEM]] 配套。

## 代表工作
- [[LeWM]]：latent MPC + CEM
- [[Dreamer]]：world model + actor 而非显式 MPC

## 相关概念
- [[Cross-Entropy Method]]
- [[World Model]]
