---
type: concept
aliases: [Dynamic Sampling Policy Optimization]
---

# DAPO

## 定义
GRPO 的改进变体，通过动态调整 group sampling 的采样数量和 clip range，提升多样性并缓解训练不稳定问题。

## 数学形式
$$
\mathcal{L}_{\text{DAPO}} = \mathcal{L}_{\text{GRPO}} + \lambda_{\text{entropy}} \cdot H(\pi_\theta)
$$

附加熵正则项鼓励探索，防止策略坍塌为单一模式。

## 核心要点
1. 动态 clip range：根据当前策略与参考策略的 KL 距离自适应调整 $\epsilon$
2. 熵奖励项防止模式坍塌
3. 针对 reasoning LLM 设计，在数学/代码任务上优于标准 GRPO

## 代表工作
- [[PRISM]]: 与 GRPO、GSPO 对比的 RL 基线

## 相关概念
- [[GRPO]]
- [[GSPO]]
