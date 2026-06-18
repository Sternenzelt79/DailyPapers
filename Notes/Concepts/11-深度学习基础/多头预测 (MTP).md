---
type: concept
aliases: [MTP, Multi-Token Prediction, 多 Token 预测]
---

# 多头预测 (MTP)

## 定义

多头预测（Multi-Token Prediction，MTP）是一种语言模型训练方法：在标准下一 token 预测的基础上，**同时预测未来 $d$ 个 token**，为每个未来位置设置独立的预测头，提供更密集的梯度信号。

## 数学形式

$$
\mathcal{L}_{\text{MTP}} = \mathcal{L}_{\text{next-token}} + \frac{1}{d} \sum_{i=1}^{d} \mathcal{L}_{\text{CE}}(X_{t+i}, f_i(h_t))
$$

其中 $f_i$ 为预测未来第 $i$ 步 token 的独立头。

## 核心要点

1. **更密集梯度**：相比仅预测下一 token，MTP 提供 $O(Td)$ 梯度信号
2. **token 空间监督**：仍在 token 空间约束，中间隐状态无显式约束
3. **固定草稿长度**：用于自投机解码时草稿长度固定为 $d$，无法超出训练范围
4. **参数增量小**：相比主模型仅增加少量预测头参数

## 代表工作

- Meta（2024）提出，用于 LLaMA 大规模预训练
- [[NextLat]]: 与 MTP 对比，NextLat 在潜态空间监督，信念状态理论保证更强，投机解码加速更显著

## 相关概念

- [[投机解码]]
- [[信念状态]]
- [[自回归Transformer]]
