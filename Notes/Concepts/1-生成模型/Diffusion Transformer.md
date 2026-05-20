---
type: concept
aliases: [DiT, 扩散Transformer, Diffusion Transformer]
---

# Diffusion Transformer (DiT)

## 定义
将 Transformer 架构替代 U-Net 用于扩散模型的去噪网络，利用自注意力机制对全局上下文建模。

## 数学形式

$$
\hat{x}_0 = \epsilon_\theta(x_t, t, c)
$$

其中 $x_t$ 为噪声输入，$t$ 为扩散时间步，$c$ 为条件信息，$\epsilon_\theta$ 为 Transformer 参数化的去噪网络。

## 核心要点
1. 用 Transformer block 替代 U-Net 的卷积结构，天然支持可变长序列
2. 通过 adaLN（自适应 LayerNorm）注入时间步和类别条件
3. 在机器人策略学习中常用于动作块（Action Chunk）的生成

## 代表工作
- [[RoVLA]]: 使用 32 层 DiT 作为低层动作生成器
- [[pi0]]: 流匹配 + DiT 骨干

## 相关概念
- [[条件流匹配]]
- [[VLA]]
