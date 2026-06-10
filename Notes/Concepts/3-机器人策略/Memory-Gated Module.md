---
type: concept
aliases: [记忆门控模块, 门控记忆, Memory-Gated Module, Memory Gate]
---

# Memory-Gated Module

## 定义

通过可学习的门控机制（读门 + 写门）控制外部记忆的读取和写入时机，在预测到的技能边界处稀疏更新记忆，避免密集历史聚合，实现事件驱动的任务状态追踪。

## 数学形式

**读门**：
$$
c^m_t = \text{Attn}(W_q x_t, W_k M_t, W_v M_t), \quad
\tilde{x}_t = x_t + \alpha^r_t W_m c^m_t
$$

**写门**：
$$
\alpha^w_t = \sigma(G_w(\tilde{x}_t, \hat{z}^h_t, \hat{b}_t)), \quad
M_{t+1} = \begin{cases} U_\psi(M_t, \gamma_t) & \alpha^w_t > \eta \\ M_t & \text{otherwise} \end{cases}
$$

## 核心要点

1. **事件驱动**: 仅在预测的技能转换边界处写入记忆，保持稀疏可解释的更新
2. **读/写解耦**: 读门和写门分别学习，读门决定从记忆中提取多少上下文，写门决定是否写入
3. **稀疏正则**: L1 正则鼓励门控值稀疏，避免过度读写
4. **因果推理**: 外部记忆仅包含历史信息，无需测试时未来预测

## 代表工作

- [[HiMem-WAM]]: 边界感知记忆门控的典型应用，用于长时域机器人操作任务的任务状态追踪

## 相关概念

- [[Boundary Scoring]]
- [[Skill Latent]]
- [[Temporal Abstraction]]
- [[World-Action Model]]
