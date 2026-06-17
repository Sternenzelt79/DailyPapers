---
type: concept
aliases: [TrXL, Transformer-XL, Gated Transformer-XL]
---

# Transformer-XL (TrXL)

## 定义
Transformer-XL 通过分段递归机制和相对位置编码，让 Transformer 能处理超过单次上下文窗口长度的序列，常用于强化学习中的长时序记忆。

## 数学形式
第 $n$ 段、第 $l$ 层的隐状态由上一段历史缓存拼接：
$$\tilde{h}_{n}^{l-1} = \text{SG}(h_{n-1}^{l-1}) \circ h_n^{l-1}$$

相对位置编码（Dai et al.）：
$$A_{i,j}^{\text{rel}} = E_{x_i}^T W_q^T W_{k,E} E_{x_j} + E_{x_i}^T W_q^T W_{k,R} R_{i-j}$$

## 核心要点
1. 分段递归：每段计算时把前段 key-value cache 拼入，梯度不跨段传播（stop-gradient）
2. 相对位置编码（XL-Net 的基础）解决绝对位置编码在跨段时的混乱
3. Gated TrXL（GTrXL）在 RL 中加了 GRU 门控，防止跨 episode 信息污染
4. 在 POMDP / 部分可观测环境中作为 memory 标准基线

## 代表工作
- Transformer-XL (Dai et al., 2019): Attentive Language Models Beyond a Fixed-Length Context
- GTrXL (Parisotto et al., 2020): Stabilizing Transformers for Reinforcement Learning

## 相关概念
- [[LSTM]]
- [[Mamba]]
