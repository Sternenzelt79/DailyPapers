---
type: concept
aliases: [知识绝缘, KI, knowledge insulation]
---

# Knowledge Insulation

## 定义

在多任务或两阶段训练中，通过 stop-gradient 操作阻断新任务（如 Flow Matching）的梯度流向预训练模块（如 VLM 前缀），防止新训练目标覆写已有知识，同时保持联合优化的灵活性。

## 数学形式

$$
\widetilde{H}_{\phi,p}^{KI} = \text{sg}(H_{\phi,p}^{KI})
$$

联合训练目标：

$$
\mathcal{L}_{KI} = \alpha \mathcal{L}_{FM} + \mathcal{L}_{FAST} + \sum_j \lambda_j \mathcal{L}^{(j)}_{CE}, \quad \alpha = 10
$$

## 核心要点

1. **动机**: Flow Matching 的 MSE 梯度若直接回传 VLM，会破坏 VLM 的语言理解能力
2. **实现**: 对 VLM 前缀隐状态 $H_{\phi,p}$ 施加 stop-gradient，梯度仅在 DiT 动作专家内传播
3. **效果**: 在保持 VLM 知识完整性的同时，DiT 能充分利用 VLM 的高级语义表示

## 代表工作

- [[LabVLA]]: 首次提出用于 VLA 两阶段训练中保护 VLM 知识

## 相关概念

- [[Flow Matching]]
- [[FAST]]
- [[DiT]]
- [[VLM]]
