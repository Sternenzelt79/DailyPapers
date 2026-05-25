---
type: concept
aliases: [Chain-of-Thought, CoT Prompting, 思维链]
---

# CoT (Chain-of-Thought)

## 定义
Chain-of-Thought，一种通过在输入中加入中间推理步骤示例，引导语言模型逐步推理后再给出最终答案的 prompting 策略；也可通过 SFT 将推理链注入模型参数。

## 数学形式
$$P(a \mid x) = P(a \mid c_1, c_2, \ldots, c_k, x)$$

其中 $c_i$ 为推理链中间步骤，$x$ 为输入，$a$ 为最终答案。

## 核心要点
1. 通过显式中间步骤提升复杂推理任务的准确性
2. Few-shot CoT：在 prompt 中提供推理示例；Zero-shot CoT：直接用"Let's think step by step"触发
3. 在 VLA 中用于生成动作前的语言推理，提升任务泛化
4. 与 self-improvement 结合时，A2L 模块产生的语言描述可以作为 CoT 反馈

## 代表工作
- [[LACY]]: 用 CoT 推理增强 VLA 的语言-动作循环，提升操作泛化

## 相关概念
- [[VLM]]
- [[Vision-Language-Action Model]]
