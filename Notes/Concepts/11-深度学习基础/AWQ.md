---
type: concept
aliases: [Activation-aware Weight Quantization, 激活感知权重量化]
---

# AWQ

## 定义
AWQ（Activation-aware Weight Quantization）是一种针对 LLM 的权重量化方法，通过保护激活值较大对应的显著权重通道来减少量化误差。

## 核心要点
1. 发现权重的重要性与对应激活值的大小高度相关
2. 对重要权重通道做 per-channel scaling（不量化），其余压缩到 INT4/INT3
3. 无需梯度，纯数据驱动，量化速度快

## 代表工作
- [[AWQ]]: Lin et al., 2023, MIT Han Lab

## 相关概念
- [[GPTQ]]
- [[QLoRA]]
