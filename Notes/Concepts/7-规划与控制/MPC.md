---
type: concept
aliases: [Model Predictive Control, 模型预测控制]
---

# MPC

## 定义
模型预测控制（Model Predictive Control）：在每个时间步，基于当前状态和系统动力学模型，在有限预测窗口内求解最优控制序列，只执行第一步动作，然后滚动重规划。

## 数学形式
$$
\min_{u_0, \ldots, u_{N-1}} \sum_{k=0}^{N-1} \ell(x_k, u_k) + \ell_f(x_N)
$$
$$
\text{s.t.} \quad x_{k+1} = f(x_k, u_k), \quad x_k \in \mathcal{X}, \quad u_k \in \mathcal{U}
$$

## 核心要点
1. 滚动优化（receding horizon）：每步重新求解，以最新观测修正计划
2. 需要显式的动力学模型 $f$（可以是真实物理模型或学到的神经网络）
3. 约束处理天然：状态约束 $\mathcal{X}$、控制约束 $\mathcal{U}$ 直接编入优化
4. 在 world model 文献中，MPC 常与学到的隐空间模型结合（MBRL + MPC）

## 代表工作
- [[PLDM]]: 用 latent 动力学模型做 MPC 的 world model 规划
- [[stable-worldmodel]]: SWM 生态中 MPC 作为标准规划算法之一

## 相关概念
- [[PLDM]]
- [[PPO]]
- [[状态空间模型]]
