---
concept: KV-Cache Conditioning
category: Architecture
tags: [vla, attention, efficient-inference, action-expert]
created: 2026-05-09
---

# KV-Cache Conditioning

## 定义

一种把流匹配 / 扩散动作专家接入 VLM 的高效条件化方案：动作专家的每一层通过 [[Cross-Attention|交叉注意力]] attend 到 VLM 同层的 KV 缓存（经线性投影对齐），而不是把动作 token 拼接到 VLM 输入序列里。

## 核心要点

1. **VLM 只前向一次**：观测 + 语言指令产生一次 KV 缓存，之后所有流匹配采样步、动作 token 生成都复用这份缓存。
2. **逐层注入**：动作专家第 $\ell$ 层只 attend VLM 第 $\ell$ 层的 $\tilde K_\ell, \tilde V_\ell$，保留多尺度信息。
3. **线性 adapter 对齐维度**：

$$
\tilde K_\ell = \text{reshape}(P_K K^{\text{vlm}}_\ell), \qquad \tilde V_\ell = \text{reshape}(P_V V^{\text{vlm}}_\ell)
$$

4. **可学习 gate**：每个交叉注意力分支带 gate $g^{\text{ca}}_\ell$，训练初期接近恒等。
5. 与 [[Pi0]] / [[Pi05]] 把噪声动作 token 拼到 VLM 序列的方案相比，推理延迟显著降低。

## 代表工作

- [[MolmoAct2]]: 首次系统化使用此方案，36 层动作专家逐层 attend Molmo2-ER KV 缓存。

## 相关概念

- [[Flow Matching]]
- [[Cross-Attention]]
- [[AdaRMS]]
- [[Action Chunking]]
