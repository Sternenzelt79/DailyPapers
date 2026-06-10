---
type: concept
aliases: [LLaMA-2, LLaMA-7B, Large Language Model Meta AI]
---

# LLaMA（Large Language Model Meta AI）

## 定义

LLaMA 是 Meta AI 开发的开源大语言模型系列，以**高效训练**和**开源可复现**为特点，是学术和工业界 VLA/多模态系统中最广泛使用的语言骨干之一。

## 数学形式

标准自回归 Transformer 语言建模：

$$
p(x) = \prod_{i=1}^{T} p(x_i | x_{<i})
$$

采用 RoPE 位置编码、SwiGLU 激活、RMSNorm 归一化。

## 核心要点

1. **LLaMA-2 7B/13B/70B**：三个尺寸，VLA 中常用 7B；
2. **RoPE（Rotary Position Embedding）**：支持外推至更长序列；
3. **在 VLA 中的角色**：作为语言理解和指令跟随的骨干，生成高层认知 token；
4. **Prismatic VLM 中的使用**：结合 DINOv2 + SigLIP 视觉编码器组成多模态 VLM。

## 代表工作

- [[MemoryVLApp]]: 7B Prismatic VLM 中使用 LLaMA-7B 生成认知 token
- [[CogACT]]: 同样基于 7B VLM + LLaMA 骨干

## 相关概念

- [[VLM]]
- [[VLA]]
- [[Diffusion Transformer]]
