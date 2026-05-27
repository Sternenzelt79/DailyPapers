---
type: concept
aliases: [DPO, Direct Preference Optimization, 直接偏好优化]
---

# DPO (Direct Preference Optimization)

## 定义
DPO 是一种从人类偏好数据中对齐语言/生成模型的方法，通过重新参数化 RLHF 目标，将 reward model 隐式地编码进策略本身，从而绕过显式的 RL 训练循环。

## 数学形式
$$\mathcal{L}_\text{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right) \right]$$

其中 $y_w$ 为偏好样本，$y_l$ 为非偏好样本，$\beta$ 控制 KL 散度约束强度。

## 核心要点
1. 不需要单独训练 reward model，直接从 pairwise 偏好数据更新策略
2. 基于 pairwise 比较，天然限制只能利用二元"好/坏"信号
3. 扩展方向：KTO（单样本）、MaPO（多样本）、DSPO（listwise）

## 在扩散模型中的应用
扩散模型 DPO（如 Diffusion-DPO）将偏好优化应用于 T2I 模型，但 pairwise 假设在多候选场景下信息利用率低——这是 DSPO（Listwise）方法提出的动机。

## 代表工作
- Rafailov et al. (2023): DPO 原始论文
- [[DSPO]]: listwise 扩展，超越 pairwise 限制
- [[GRPO]]: RL 路线（与 DPO 并列的对齐方法）

## 相关概念
- [[RLHF]]
- [[GRPO]]
- [[PPO]]
