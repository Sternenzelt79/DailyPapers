---
type: concept
aliases: [AR Transformer, Autoregressive Transformer, 自回归变换器, 因果Transformer]
---

# 自回归 Transformer（AR Transformer）

## 定义

自回归 Transformer 是使用因果注意力的 Transformer 架构，按从左到右的顺序逐步生成序列，每个位置的输出仅依赖于之前的输入和输出，是 GPT 系列和现代具身 AI 骨干的核心结构。

## 数学形式

$$
p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1})
$$

## 核心要点

1. 因果掩码确保自回归约束（不看未来）
2. 支持多模态混合序列：文本 token + 图像 patch + 动作 token 统一建模
3. 相比双向 Transformer（BERT）更适合序列生成，相比扩散 Transformer 推理延迟更低
4. WLA 选择 AR Transformer 而非扩散 Transformer，实现统一的三流输出

## 代表工作

- [[WLA]]: 以 RynnBrain-2B 为 AR Transformer 骨干，统一预测文本子任务、物理动态和动作

## 相关概念

- [[因果注意力]]
- [[大语言模型]]
- [[自回归策略]]
- [[Meta-Query]]
