---
type: concept
aliases: [教师强制记忆预热, 教师强制预热, Teacher-Forced Memory Warmup]
---

# Teacher-Forced Memory Warmup

## 定义

在涉及记忆模块的端到端训练初期，用真实的（Teacher-Forced）边界标签而非模型预测的边界来驱动记忆写入，从而稳定记忆模块的初始训练，避免早期预测边界噪声导致的训练崩溃。

## 数学形式

$$
M^{\text{teach}}_{t+1} = \begin{cases} U_\psi(M^{\text{teach}}_t, \gamma^{\text{teach}}_t) & \text{if } \bar{b}_t = 1 \\ M^{\text{teach}}_t & \text{otherwise} \end{cases}
$$

其中 $\bar{b}_t$ 为由 Stage II 发现并映射回时间步的真实边界标签，而非 Stage III 中预测的 $\hat{b}_t$。

## 核心要点

1. **训练稳定**: 避免了早期预测边界不准确导致记忆内容混乱进而影响整体训练
2. **渐进过渡**: 预热阶段结束后，逐渐切换为模型预测的边界驱动记忆（Scheduled Sampling 变体）
3. **与 Teacher Forcing 的关系**: 传统 Teacher Forcing 是对序列解码器用真实前一步输出；这里是对记忆写入用真实边界
4. **依赖前序阶段**: 需要 Stage II 的边界发现结果作为真实标签，体现多阶段训练的设计依赖

## 代表工作

- [[HiMem-WAM]]: Stage III 微调初期使用教师强制记忆预热，稳定门控记忆模块的训练

## 相关概念

- [[Memory-Gated Module]]
- [[Boundary Scoring]]
- [[Skill Latent]]
