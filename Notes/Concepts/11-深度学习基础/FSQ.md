---
type: concept
aliases: [Finite Scalar Quantization]
---

# FSQ

## 定义
Finite Scalar Quantization，一种将连续向量量化为有限离散码本的方法，相比 VQ 更稳定（无 codebook collapse 问题）；在 SONIC 中用于 motion tokenization。

## 核心要点
1. 每个维度独立量化到有限整数集合
2. 无需维护 codebook，避免 VQ 的 collapse 问题
3. 可微，适合端到端训练

## 相关概念
- [[VQ]]
- [[SONIC]]
