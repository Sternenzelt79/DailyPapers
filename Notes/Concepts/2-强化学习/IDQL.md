---
type: concept
aliases: [Implicit Diffusion Q-Learning]
---

# IDQL

## 定义
IDQL（Implicit Diffusion Q-Learning）将扩散模型作为策略表示，通过在推理时用 Q 函数对扩散模型生成的动作候选做重采样，实现 offline RL 策略提升，无需反向传播过扩散过程。

## 数学形式
$$a^* = \arg\max_{a \sim \pi_{\text{diffusion}}(\cdot|s)} Q(s, a)$$

## 核心要点
1. 扩散策略作为 behavior policy，生成多个 action candidates
2. Q-guided resampling 选择最优动作，不改变扩散模型本身
3. 解耦了 diffusion 和 Q-learning，训练稳定

## 代表工作
- Hansen-Estruch et al. (2023), IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies

## 相关概念
- [[AWR]]
- [[FQL]]
- [[Diffusion Policy]]
