---
type: concept
aliases: [CNN-Transformer混合架构, Hybrid CNN-Transformer]
---

# CNN-Transformer

## 定义

结合卷积神经网络（CNN）的局部特征提取能力与 Transformer 的全局注意力机制的混合架构，常用于图像/传感器信号的紧凑表示学习。

## 核心要点

1. **CNN 分支**: 提取局部纹理和空间结构特征，计算高效
2. **Transformer 分支**: 通过自注意力捕获长程依赖关系
3. **融合策略**: 可串联（CNN 提取特征后输入 Transformer）或并联后融合

## 代表工作

- [[TacForeSight]]: 触觉分词器采用 CNN-Transformer 将触觉图像编码为紧凑 token

## 相关概念

- [[Transformer]]
- [[Cross-Attention]]
