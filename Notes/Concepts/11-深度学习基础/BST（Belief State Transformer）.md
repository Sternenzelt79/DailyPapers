---
type: concept
aliases: [BST, Belief State Transformer, 信念状态变换器]
---

# BST（Belief State Transformer）

## 定义

BST（Belief State Transformer）是一种通过**前向-后向计算**显式学习信念状态的 Transformer 变体：前向 Transformer 处理历史信息，后向 Transformer 处理未来信息，两者共同生成充分统计量（信念状态）表示。

## 核心要点

1. **显式信念状态**：同时利用过去和未来 token，使隐状态包含双向信息
2. **计算复杂度 $O(T^2)$**：需要前向和后向两次完整扫描，序列长度平方代价
3. **参数量翻倍**：约 2.57B（vs GPT 1.32B），训练速度仅为 GPT 的 29%
4. **理论完备**：但计算代价使其在大规模应用中不实用

## 代表工作

- [[NextLat]]: 以 $O(Td)$ 代价实现等价的信念状态收敛保证，无需 BST 的二次代价

## 相关概念

- [[信念状态]]
- [[自回归Transformer]]
- [[GPT]]
