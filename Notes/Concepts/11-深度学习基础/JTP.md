---
type: concept
aliases: [Joint Token Prediction, 联合 Token 预测]
---

# JTP

## 定义

JTP（Joint Token Prediction）是一种在 token 空间**联合预测**当前和多个未来 token 的训练方法，通过在同一前向传播中同时生成多个 token 的概率分布来提供前瞻性梯度信号。

## 核心要点

1. **联合预测**：在单次前向传播中联合建模 $P(X_{t+1}, X_{t+2}, \ldots, X_{t+d} \mid X_{1:t})$
2. **仍在 token 空间**：与 MTP 类似，监督信号在离散 token 层面，不直接约束中间隐状态
3. **无信念状态保证**：不能从理论上保证隐状态收敛到信念状态
4. **投机解码有限**：草稿质量受限于 token 空间预测精度

## 代表工作

- [[NextLat]]: 对比 JTP，NextLat 在潜态空间监督，Countdown 准确率超出 >12 个百分点，投机解码加速 2× 以上

## 相关概念

- [[多头预测 (MTP)]]
- [[信念状态]]
- [[投机解码]]
- [[GPT]]
