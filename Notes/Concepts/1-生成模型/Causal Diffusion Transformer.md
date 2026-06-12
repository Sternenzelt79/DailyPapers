---
type: concept
aliases: [Causal DiT, 因果扩散 Transformer, 块因果扩散模型]
---

# Causal Diffusion Transformer

## 定义

将扩散/Flow Matching 生成目标与块因果（block-causal）自回归掩码结合的 Transformer 架构，可按时间顺序自回归生成视觉或视觉-动作序列。

## 数学形式

块因果掩码允许当前块只关注历史块：

$$
u_{t:t+k} = [z_{t:t+k},\, \ell_{t:t+k-1}]
$$

Flow Matching 联合目标：

$$
\mathcal{L}_{FM} = \mathbb{E}\left[\|F_\theta^v(x_\alpha, \alpha, s_{<t}) - \dot{x}_\alpha^v\|_2^2 + \lambda_a \|F_\theta^a(x_\alpha, \alpha, s_{<t}) - \dot{x}_\alpha^a\|_2^2\right]
$$

## 核心要点

1. 块因果 masking：同一时间块内全注意力，跨块单向（只看历史）
2. 视觉和动作 token 共享注意力权重，但使用模态独立的前馈网络（FFN）
3. 语言指令通过冻结文本编码器嵌入为全局条件 token
4. 可与 Flow Matching 或 DDPM 配合作为生成目标

## 代表工作

- [[RepWAM]]: 1.3B / 5B Causal DiT，联合预测视觉 chunk 和潜在动作 chunk

## 相关概念

- [[Flow Matching]]
- [[Diffusion Transformer]]
- [[World-Action Model]]
- [[Latent Action]]
