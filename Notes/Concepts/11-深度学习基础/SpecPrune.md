---
type: concept
aliases: [Spectral Pruning]
---

# SpecPrune

## 定义
基于频谱分析的模型剪枝方法，通过分析权重矩阵的奇异值分布识别冗余的注意力头或层，实现结构化压缩。

## 数学形式
$$\text{importance}(l) = \frac{\sigma_1(W_l)}{\sum_i \sigma_i(W_l)}$$

其中 $\sigma_i$ 为奇异值，通过低秩近似决定保留哪些层/头。

## 核心要点
1. 无需梯度或微调，属于 training-free 剪枝
2. 对比 [[CKA]] 方法，SpecPrune 关注权重谱而非激活相似度
3. 在 VLA 压缩领域作为基线对比方法

## 代表工作
- [[CLP]]: 对比基线

## 相关概念
- [[CKA]]
- [[MoLe]]
