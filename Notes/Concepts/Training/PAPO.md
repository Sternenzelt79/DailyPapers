---
type: concept
aliases: [Planning-Aware Policy Optimization, PAPO-VLA]
---

# PAPO

## 定义

PAPO（Planning-Aware Policy Optimization）是一种规划感知策略优化方法，通过识别关键规划动作并利用因果充分性与必要性衡量其重要度，将差异化权重融入 [[GRPO]] 优势估计，提升 [[Vision-Language-Action Model|VLA]] 策略在机器人操作任务中的可靠性。

## 数学形式

规划感知优势估计：

$$\widehat{A}_{i,t} = A_{i,t} + \eta \cdot C_{i,t}^{plan}$$

规划感知优化目标（替换标准 [[GRPO]] 中的均匀优势）：

$$J_{PAPO}(\theta) = \mathbb{E} \left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=0}^{T_i-1}\left(\min(\rho_{i,t}\widehat{A}_{i,t},\; \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\widehat{A}_{i,t}) - \beta D_{KL}\right)\right]$$

## 核心要点

1. **规划动作识别**: 结合动作变化幅度与结果门控 $s_t = \tilde{u}_t \cdot g(\tau)$，选取 top-k 时间步
2. **因果重要度**: [[因果充分性]] 和 [[因果必要性]] 的调和平均 $C_t^{plan}$
3. **差异化权重**: 关键规划时刻获得更强梯度信号，执行动作权重保持标准水平
4. **最优超参**: $\eta = 0.15$（LIBERO benchmark）

## 代表工作

- [[PAPO-VLA]]: 提出并验证 PAPO 方法

## 相关概念

- [[GRPO]]
- [[因果充分性]]
- [[因果必要性]]
- [[因果重要度]]
- [[Vision-Language-Action Model]]
