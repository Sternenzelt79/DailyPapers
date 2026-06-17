---
type: concept
aliases: [Diffusion Policy PPO, 扩散策略强化学习]
---

# DPPO

## 定义
将扩散策略（Diffusion Policy）与近端策略优化（PPO）结合的在线强化学习框架，通过 PPO 目标微调预训练扩散策略。

## 核心要点
1. 预训练扩散策略作为初始化，PPO 在线微调避免从零训练
2. 扩散的多步去噪过程被视为序列决策，每步去噪视为一个动作
3. 通过 ELBO 技巧将扩散 score 匹配损失与 PPO surrogate 目标结合
4. 优势在于保留扩散策略的多模态行为分布，同时获得 RL 的探索能力

## 相关概念
- [[Diffusion Policy]]
- [[PPO]]
- [[Action Chunking]]
