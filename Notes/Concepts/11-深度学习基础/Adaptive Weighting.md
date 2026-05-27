---
type: concept
aliases: [自适应权重调度, 动态权重, Adaptive Loss Weighting]
---

# Adaptive Weighting

## 定义

在多任务或多损失训练中，动态调整各损失项权重的机制，使模型在训练不同阶段能自动平衡主任务与辅助任务的优化优先级。

## 数学形式

基于 EMA 的自适应权重（RoVLA 方案）：

$$
\lambda_j = \frac{1}{1 + \mathcal{L}_j^{ema}}
$$

$$
\mathcal{L}_j^{ema} = \gamma \mathcal{L}_{j-1}^{ema} + (1 - \gamma) \mathcal{L}_{\text{main}}^{(j)}
$$

其中 $\gamma$ 为 EMA 衰减系数，初始 $\mathcal{L}_0^{ema}$ 设为较大值（如 100）使初始 $\lambda$ 接近 0。

## 核心要点

1. 训练初期主损失较大，$\lambda \approx 0$，辅助约束权重低，优先学习主任务
2. 随训练进行主损失下降，$\lambda$ 逐渐增大，辅助约束比重上升
3. EMA 平滑了损失波动对权重的影响，比直接用瞬时损失更稳定
4. 与固定权重相比，无需手动调参 $\lambda$

## 代表工作

- [[RoVLA]]: 用主监督损失的 EMA 自适应调节一致性约束权重，使模型先学动作再学鲁棒性

## 相关概念

- [[Evolutionary Consistency]]
- [[EMA]]
- [[联合损失]]
