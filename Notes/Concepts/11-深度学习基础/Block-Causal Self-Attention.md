---
type: concept
aliases: [块因果自注意力, 块因果注意力, Block-Causal Attention]
---

# Block-Causal Self-Attention

## 定义

块因果自注意力是一种因果注意力变体，以"块"（block）为粒度施加因果掩码：同一块内的 token 可相互 attend，跨块则只能 attend 到更早的块，从而在时序预测中防止未来信息泄露。

## 数学形式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M_{\text{block-causal}}\right) V
$$

其中掩码 $M_{\text{block-causal}}[i,j] = 0$ 当 $\text{block}(j) \leq \text{block}(i)$，否则 $= -\infty$。

## 核心要点

1. **块内全连接**: 同一时间步（块）内的所有 token（多视角 patch + 辅助 token）可自由交互
2. **跨块单向依赖**: $t'$ 时刻的块只能 attend 到 $t'' \leq t'$ 的块，保证自回归预测的因果性
3. **多视角扩展**: 在机器人策略中，每个时间步的"块"包含多视角几何 token + 本体感知 + 动作 token

## 代表工作

- [[GAM]]: 在因果未来预测器中使用块因果自注意力预测未来几何潜在 token

## 相关概念

- [[Transformer]]
- [[时序世界建模]]
- [[World-Action Model]]
