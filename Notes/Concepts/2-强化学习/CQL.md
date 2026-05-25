---
type: concept
aliases: [Conservative Q-Learning, CQL]
---

# CQL

## 定义
Conservative Q-Learning，一种 offline RL 算法，通过在值函数优化中加入正则项压低分布外（OOD）动作的 Q 值估计，避免 offline RL 中的 value overestimation 问题。

## 数学形式
$$\mathcal{L}_{\text{CQL}}(\theta) = \alpha \cdot \mathbb{E}_{s \sim \mathcal{D}}\left[\log \sum_a \exp(Q_\theta(s,a)) - \mathbb{E}_{a \sim \mathcal{D}}[Q_\theta(s,a)]\right] + \frac{1}{2}\mathbb{E}_{(s,a,s') \sim \mathcal{D}}\left[(Q_\theta(s,a) - \hat{B}^\pi Q_\theta(s,a))^2\right]$$

## 核心要点
1. 在标准 Bellman 备份基础上加保守正则项，压低 OOD 动作 Q 值
2. 提升在分布外动作上的悲观估计，防止策略利用 Q 值估计漏洞
3. 支持 offline RL 在无需与环境交互的情况下学习有效策略
4. 在 few-shot robot manipulation adaptation 中被用于训练 value network

## 代表工作
- [[VGAS]]: 基于 CQL 训练 value network 为 VLA action chunk 候选打分，实现 few-shot 适应

## 相关概念
- [[Soft Actor-Critic]]
- [[Diffusion Policy]]
- [[Action Chunking]]
