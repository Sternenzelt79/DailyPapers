---
type: concept
aliases: [AdaLN, Adaptive LayerNorm, Adaptive Layer Normalization]
---

# Adaptive Layer Normalization (AdaLN)

## 定义
一种条件归一化层：用外部条件向量（如时间步、动作、类别）回归出 LayerNorm 的 scale $\gamma$ 与 shift $\beta$，按 token 注入，而不是把条件向量拼到 token 序列里。

## 数学形式
$$
\mathrm{AdaLN}(x, c) = \gamma(c) \cdot \mathrm{LN}(x) + \beta(c)
$$
其中 $\gamma, \beta$ 是 $c$ 的小型 MLP 输出。

## 核心要点
1. 比 cross-attention 注入条件更省参数、更稳；常见于 [[DiT]]、扩散模型与控制类 transformer。
2. 在动作条件世界模型里常用来把 $a_t$ 注入预测器每一层。
3. 变种：AdaLN-Zero（初始化时 $\gamma=0,\beta=0$），让残差路径起步像恒等。

## 代表工作
- [[LeWM]]：predictor 用 AdaLN 注入动作 $a_t$
- DiT (Peebles & Xie, 2023)：原始提出于 diffusion transformer

## 相关概念
- [[Layer Normalization]]
- [[FiLM]]
