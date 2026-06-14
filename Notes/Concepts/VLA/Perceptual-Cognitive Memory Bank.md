---
type: concept
aliases: [PCMB, 感知-认知记忆库, 感知认知记忆]
---

# Perceptual-Cognitive Memory Bank（PCMB）

## 定义

PCMB 是 MemoryVLA++ 提出的**双流历史记忆模块**，同步存储细粒度感知 token 和高层认知 token，通过跨注意力检索 + 门控融合 + 冗余感知整合，为 VLA 策略提供时序上下文。

## 数学形式

记忆库定义：

$$
M_{pcmb} = \{m^x | x \in \{p, c\}\}, \quad m^x = \{m^x_i \in \mathbb{R}^{N_x \times d_x}\}_{i=1}^{L}
$$

记忆检索（跨注意力）：

$$
\hat{H}^x = \operatorname{softmax}\!\left(\frac{q^x (K^x)^\top}{\sqrt{d_x}}\right) V^x
$$

门控融合：

$$
g^x = \sigma(\operatorname{MLP}(\operatorname{concat}[x, H^x])), \quad \tilde{x} = g^x \odot H^x + (1-g^x) \odot x
$$

冗余感知整合：

$$
i^*_x = \arg\max_{i=1,\ldots,L-1} \cos(m^x_i, m^x_{i+1}), \quad m^x_{i^*_x} \leftarrow \tfrac{1}{2}(m^x_{i^*_x} + m^x_{i^*_x+1})
$$

## 核心要点

1. **双流设计**：感知 token 保留空间细节，认知 token 提供语义，缺一不可；
2. **冗余感知整合（token merge）**：替代 FIFO，通过合并相似相邻条目保留更长时间跨度的历史；
3. **时间步位置编码**：区分不同历史时刻的上下文；
4. **记忆引导想象**：PCMB 的输出同时用于过滤世界模型的想象 token。

## 代表工作

- [[MemoryVLApp]]: PCMB 的提出论文，L=16 默认容量

## 相关概念

- [[Cross-Attention]]
- [[MemoryVLA]]
- [[世界模型]]
- [[VLA]]
