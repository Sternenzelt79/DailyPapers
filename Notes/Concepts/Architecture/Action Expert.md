---
concept: Action Expert
aliases: [Action Head, Action Decoder]
category: Architecture
tags: [vla, action-decoder, cross-attention]
created: 2026-05-09
---

# Action Expert

## 定义

Action Expert 是 [[Vision-Language-Action Model|VLA]] 中负责把 VLM 表征解码为机器人动作的小型 Transformer。常通过 [[Cross-Attention]] 读取 VLM 的 KV 缓存，避免重算主干。

## 典型设计 (MolmoAct2 版)

每层包含三个残差块（带门控）：

$$
h'_\ell = h_\ell + g^{sa}_\ell \cdot \text{SA}(h_\ell)
$$

$$
\bar{h}_\ell = h'_\ell + g^{ca}_\ell \cdot \text{CA}(Q_\ell, \tilde{K}_\ell, \tilde{V}_\ell)
$$

$$
h_{\ell+1} = \bar{h}_\ell + g^{ff}_\ell \cdot \text{MLP}(\bar{h}_\ell)
$$

其中 $\tilde{K}_\ell = \text{reshape}(P_K K^{vlm}_\ell)$ 是 VLM KV 的投影。

## 训练目标

通常配合 [[Flow Matching]] 损失：

$$
\mathcal{L}_{\text{flow}} = \mathbb{E}\,\| m \odot (f_\theta(x_t, t, c) - u) \|_2^2
$$

## 代表工作

- [[Pi0]] / [[Pi05]]: 早期 Flow 匹配 Action Expert。
- [[MolmoAct2]] (2026): 加入 KV 投影 + 三处门控。

## 相关概念

- [[Cross-Attention]]
- [[Flow Matching]]
- [[Action Chunking]]
