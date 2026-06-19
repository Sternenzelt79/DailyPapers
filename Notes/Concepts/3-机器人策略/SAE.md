---
type: concept
aliases: [Structured Action Expert, 结构化动作专家]
---

# SAE（Structured Action Expert）

## 定义

一种替换标准 VLA 单体动作头的模块化结构，将双臂动作分解为共享潜变量（协调意图）和臂特定残差潜变量（个体调整），并通过任务自适应辅助损失塑造协调结构。

## 数学形式

$$
z^s_t = g_s(h_t), \quad z^L_t = g_L(h_t), \quad z^R_t = g_R(h_t)
$$

$$
a_{t,L} = \phi^s_L(z^s_t) + \phi^r_L(z^L_t), \quad a_{t,R} = \phi^s_R(z^s_t) + \phi^r_R(z^R_t)
$$

## 核心要点

1. 三个并行编码器生成一个共享潜变量和两个残差潜变量
2. 任务自适应损失：对称任务用 $\mathcal{L}_{\text{sparse}}$，非对称任务用 $\mathcal{L}_{\text{shared}}$，时序耦合任务用 $\mathcal{L}_{\text{sync}}$
3. 兼容任意连续动作 VLA 骨干，无需修改主干网络
4. 训练分两阶段：热身阶段冻结骨干仅训练 SAE，再全量微调

## 代表工作

- [[Co-VLA]]: SAE 的提出工作，基于 [[π0]] 骨干在双臂操作中验证

## 相关概念

- [[共享-残差分解]]
- [[LAC|Latent-Aware Controller]]
- [[π0]]
