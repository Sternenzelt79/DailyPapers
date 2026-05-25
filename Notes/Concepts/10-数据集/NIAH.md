---
type: concept
aliases: [Needle in a Haystack, 大海捞针测试, NIAH-1, NIAH-2, NIAH-3]
---

# NIAH

## 定义
Needle in a Haystack（大海捞针），长上下文语言模型评估基准，测试模型在超长文本中定位和提取特定信息片段（"needle"）的能力，常以不同位置深度和上下文长度绘制热力图。

## 数学形式
$$\text{score} = \mathbb{1}[\hat{y} = y^*], \quad (d, L) \in [0,1] \times \{1K, 2K, \ldots, 128K\}$$

其中 $d$ 为 needle 位置深度，$L$ 为上下文长度。

## 核心要点
1. **NIAH-1**：单 needle，单问题，基础长上下文检索
2. **NIAH-2**：多 needle，多问题，需同时提取多个信息
3. **NIAH-3**：多 needle 聚合，需跨位置综合多条信息推理
4. 常配合 [[RULER]] 使用构成综合长上下文评测套件

## 代表工作
- [[GatedDeltaNet-2]]: 在 NIAH-1/2/3 上验证线性注意力机制的长上下文记忆能力

## 相关概念
- [[Transformer]]
- [[滑动窗口注意力]]
