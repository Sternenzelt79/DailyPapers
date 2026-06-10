---
type: concept
aliases: [Two-Tower Mixture-of-Transformers, 双塔MoT, AR-DM双塔架构]
---

# Two-Tower MoT

## 定义
一种将自回归（AR）Transformer 和扩散（DM）Transformer 融合在同一前向传播中的架构，AR Tower 处理离散 token 负责推理，DM Tower 处理连续 token 负责生成，二者通过 joint attention 共享信息。

## 数学形式

在同一 Transformer 层中，AR 和 DM token 使用各自的参数集计算，但注意力 K/V 来自合并序列：

$$
H_{AR} = \text{Attn}_{\theta_{AR}}(Q_{AR},\; [K_{AR}; K_{DM}],\; [V_{AR}; V_{DM}]) + \text{FFN}_{\theta_{AR}}
$$

$$
H_{DM} = \text{Attn}_{\theta_{DM}}(Q_{DM},\; [K_{AR}; K_{DM}],\; [V_{AR}; V_{DM}]) + \text{FFN}_{\theta_{DM}}
$$

## 核心要点
1. **双塔独立参数**: AR Tower（因果注意力，适合推理）和 DM Tower（全注意力，适合生成）使用完全独立的参数
2. **Joint Attention**: 两个 tower 的 token 在注意力中互相可见，推理结果直接条件化生成
3. **统一前向传播**: 两个 tower 在同一 forward pass 中计算，而非顺序串联，效率更高
4. **单/双塔模式**: Reasoner 模式仅激活 AR Tower，Generator 模式同时激活两塔

## 代表工作
- [[Cosmos3]]: Two-Tower MoT 的提出者，实现了 omnimodal world model

## 相关概念
- [[Mixture-of-Transformers]]
- [[自回归变换器]]
- [[Diffusion Transformer]]
- [[扩散世界模型]]
