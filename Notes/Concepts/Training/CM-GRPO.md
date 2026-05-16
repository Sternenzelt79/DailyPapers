---
type: concept
aliases: [CM-GRPO, Consistency Model GRPO]
---

# CM-GRPO

## 定义
CM-GRPO 是将 Consistency Model（一致性模型）与 GRPO（Group Relative Policy Optimization）结合的在线强化学习方法，用于对齐自回归视频生成模型的训练分布与推理分布。

## 数学形式
$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}_{x \sim p_\theta} \left[ r(x) - \beta \cdot D_{\text{KL}}(p_\theta \| p_{\text{ref}}) \right]$$

其中 $r(x)$ 是视频质量奖励，$\beta$ 控制与参考策略的偏离程度。

## 核心要点
1. Consistency Model 提供 few-step 生成能力，避免完整 ODE 求解
2. GRPO 在线采样：用模型当前的自回归 history 作为输入（而非 GT），计算奖励并优化策略
3. 解决因果自回归视频生成中的 training-inference distribution gap

## 代表工作
- [[RAVEN]]: 首次将 CM-GRPO 应用于实时自回归视频外推

## 相关概念
- [[GRPO]]（Group Relative Policy Optimization 基础算法）
- [[Diffusion Model]]（扩散模型基础）
- [[CausVid]]（因果自回归视频生成）
- [[DMD]]（Diffusion Model Distillation）
