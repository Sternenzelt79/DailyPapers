---
type: concept
aliases: [OC, Observational Consistency, 观测一致性]
---

# Observational Consistency（观测一致性）

## 定义
通过对视觉特征和机器人状态施加对抗扰动，然后约束扰动前后的速度场预测一致，提升策略对视觉/本体感知噪声的鲁棒性。

## 数学形式

$$
\mathcal{L}_{OC} = \frac{1}{2}\sum_{i=1}^{2}\left\|\hat{v}_{pert}^{\tau_i} - \mathrm{sg}\left(\hat{v}_{clean}^{\tau_i}\right)\right\|_2^2
$$

其中 $\mathrm{sg}(\cdot)$ 为 stop-gradient 算子，防止双侧预测互相塌缩。

## 核心要点
1. 利用 EC 梯度生成对抗扰动，复用已有计算图
2. stop-gradient 是防止 collapse 的关键设计
3. 扰动作用于语义视觉特征 $v_t$ 和机器人状态 $q_t$

## 代表工作
- [[RoVLA]]: OC 与 EC 协同对机器人状态和视觉噪声扰动最有效

## 相关概念
- [[一致性正则化]]
- [[Evolutionary Consistency]]
- [[对抗训练]]
