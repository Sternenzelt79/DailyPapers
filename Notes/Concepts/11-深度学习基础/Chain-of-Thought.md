---
type: concept
aliases: [思维链, CoT, Chain of Thought, Thinking]
---

# Chain-of-Thought

## 定义

一种提示（prompting）或推理策略，让语言模型在给出最终答案前先生成中间推理步骤，从而提升复杂问题的解答准确率。

## 数学形式

$$
P(y \mid x) = \sum_{r} P(y \mid x, r) \cdot P(r \mid x)
$$

其中 $r$ 为中间推理链（reasoning chain），$x$ 为输入，$y$ 为最终答案。

## 核心要点

1. **显式推理步骤**：模型在回答前先输出一系列逻辑步骤，相当于"打草稿"
2. **Thinking 模式**：部分模型（如 Gemini 2.5、Claude 3.5）支持内部 thinking token，在不可见的上下文中完成推理再输出答案
3. **提升复杂任务性能**：对需要多步逻辑推导、空间推理、数学计算的任务效果显著
4. **在机器人规划中的应用**：高层 VLM 开启 thinking 后，任务分解和子目标推理能力显著提升

## 代表工作

- [[HiVLA-Study]]: 发现 VLM thinking 能力对长时程和推理型机器人任务影响远大于模型规模

## 相关概念

- [[VLM]]
- [[Options Framework]]
