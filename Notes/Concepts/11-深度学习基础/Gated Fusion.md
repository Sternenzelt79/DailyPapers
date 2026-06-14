---
type: concept
aliases: [门控融合, Gated Mixing, Adaptive Gating]
---

# Gated Fusion

## 定义

一种通过可学习门控权重（sigmoid 输出）自适应地混合两个特征向量的融合机制，常用于将检索/生成的上下文信息与当前表征进行自适应合并。

## 数学形式

$$
g = \sigma\!\left(\text{MLP}(\text{concat}[x_1, x_2])\right)
$$

$$
y = g \odot x_1 + (1 - g) \odot x_2
$$

其中 $\sigma$ 为 Sigmoid，$g \in (0,1)^d$ 为逐元素门控权重，$\odot$ 为 Hadamard 积。

## 核心要点

1. **软选择**：相比硬选择（argmax），门控融合允许连续插值，梯度可传播
2. **内容自适应**：门控权重由两个输入的拼接动态计算，而非固定超参数
3. **防退化**：若上下文无关，门控趋向 0，退化为 $y \approx x_2$（保留原始特征），不引入噪声
4. **与残差连接的区别**：残差连接是固定 0.5/0.5 混合，门控融合是内容自适应权重

## 代表工作

- [[MemoryVLApp]]: 在 PCMB 记忆检索后用门控融合合并历史上下文与当前 token，以及想象模块中合并未来信息
- [[MemoryVLA]]: 工作记忆融合中使用门控机制
- [[CogACT]]: 扩散动作专家中使用类似门控结构

## 相关概念

- [[Cross-Attention]]: 常用于生成被融合的上下文特征
- [[Sigmoid]]: 门控权重的激活函数
- [[Adaptive Layer Normalization]]: 另一种条件自适应机制
