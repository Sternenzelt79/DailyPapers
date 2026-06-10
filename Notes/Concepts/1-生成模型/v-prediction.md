---
type: concept
aliases: [v-parameterization, velocity prediction, v参数化]
---

# v-prediction

## 定义
扩散/流匹配模型的一种参数化形式，网络直接预测速度场（velocity）$v = \epsilon - x_0$，而非噪声 $\epsilon$ 或清洁数据 $x_0$。在低噪声区间（σ→0）具有最优的梯度缩放性质。

## 数学形式

以 Flow Matching 为例，一致性函数采用 v-prediction 形式：

$$
f(x_\sigma, \sigma) = x_\sigma - \sigma \cdot v_\theta(x_\sigma, \sigma)
$$

等价于 $a(\sigma) = 1,\; b(\sigma) = -\sigma$，满足 $b'(0) = -1 \neq 0$，梯度缩放为线性 O(σ)。

## 核心要点
1. 在低噪声区间（σ→0）梯度缩放为线性 O(σ)，而标准 LCM Karras 参数化为二次 O(σ²)
2. 由 Flash-WAM Proposition 1 证明：$b'(0) \neq 0$ 是达到最优线性缩放的充要条件
3. 适用于动作流等训练质量集中于低噪声区间的模态
4. 相比 $\epsilon$-prediction 和 $x_0$-prediction，v-prediction 在高/低噪声区间均有较好数值性质

## 代表工作
- [[Flash-WAM]]: 将 v-prediction 参数化应用于动作流一致性蒸馏，解决低 σ 梯度消失问题

## 相关概念
- [[Flow Matching]]: v-prediction 是 Flow Matching 中的自然参数化选择
- [[一致性蒸馏]]: v-prediction 的线性梯度缩放性质对一致性蒸馏至关重要
- [[SNR-Shifted Scheduler]]: 决定哪个模态的训练质量集中于低噪声区间
