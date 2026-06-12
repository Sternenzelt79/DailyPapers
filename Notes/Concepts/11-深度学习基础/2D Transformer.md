---
type: concept
aliases: [2D Transformer, Spatial-Temporal Transformer]
---

# 2D Transformer

## 定义
在空间和时间两个维度上分别施加注意力的 Transformer 变体，常用于视频生成和世界模型中同时处理空间 patch token 和时序帧信息。

## 数学形式
$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

空间注意力作用于同一时间步的 patch token，时间注意力（因果）作用于同一位置的跨帧 token。

## 核心要点
1. **空间注意力**: 每帧内部的 patch 之间做自注意力，捕获空间结构
2. **因果时间注意力**: 跨帧时间维度做因果注意力，保证自回归生成
3. **常见组件**: RMSNorm、RoPE、QKNorm、SwiGLU 等现代 LLM 技术
4. **Token Dropping**: 结合 SPRINT 等方法可激进丢弃冗余 token 加速推理

## 代表工作
- [[WEAVER]]: 使用 32 层 2D Transformer 作为潜动力学模型，928M 参数

## 相关概念
- [[Transformer]]
- [[RoPE]]
- [[RMSNorm]]
- [[SwiGLU]]
- [[KV Cache]]
