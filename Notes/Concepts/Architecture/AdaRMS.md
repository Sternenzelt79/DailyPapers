---
concept: AdaRMS
category: Architecture
tags: [normalization, conditioning, diffusion, dit]
created: 2026-05-09
---

# AdaRMS

## 定义

Adaptive RMSNorm。把 [[DiT]] 中的 AdaLN（Adaptive LayerNorm）换成 RMSNorm 版本：用条件向量（如时间嵌入）线性映射出 shift / scale / gate 三个调制参数，作用在 RMSNorm 之后。

## 数学形式

$$
\text{AdaRMS}(x, c) = (1 + \alpha(c)) \cdot \text{RMSNorm}(x) + \beta(c)
$$

并附带可学习 gate $g(c)$ 控制残差强度：

$$
y = x + g(c) \cdot \text{Block}(\text{AdaRMS}(x, c))
$$

## 核心要点

1. 比 AdaLN 节省一半参数（无均值减除项）。
2. 与现代大语言模型中的 RMSNorm 兼容，便于把 DiT 思想嵌入 LLM 风格的动作专家。
3. 在流匹配动作专家中，$c$ 通常是流时间 $t$ 的正弦嵌入。

## 代表工作

- [[MolmoAct2]]: 36 层流匹配动作专家用 AdaRMS 调制时间。

## 相关概念

- [[DiT]]
- [[Flow Matching]]
- [[RMSNorm]]
