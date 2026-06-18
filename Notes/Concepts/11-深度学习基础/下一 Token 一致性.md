---
type: concept
aliases: [Next-Token Consistency, Next Token Consistency]
---

# 下一 Token 一致性

## 定义

下一 Token 一致性（Next-Token Consistency）是 NextLat 方法中对 Transformer 输出头的约束条件：模型参数化的输出分布必须与真实条件分布相等。

## 数学形式

$$
p_\theta(X_{t+1} \mid h_t) = P(X_{t+1} \mid X_{1:t})
$$

即给定当前隐状态 $h_t$，模型预测的下一 token 分布等于真实历史条件分布。

## 核心要点

1. **标准语言模型目标的精确化表述**：是 GPT 交叉熵训练目标的数学等价表达
2. **与转移一致性联合**：NextLat 的 Theorem 3.2 证明，同时满足下一 token 一致性和[[转移一致性]]时，隐状态收敛到[[信念状态]]

## 代表工作

- [[NextLat]]: Theorem 3.2 的前提条件之一

## 相关概念

- [[转移一致性]]
- [[信念状态]]
- [[自回归Transformer]]
