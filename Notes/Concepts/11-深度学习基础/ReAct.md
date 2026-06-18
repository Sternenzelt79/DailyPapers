---
type: concept
aliases: [ReAct, Reason+Act, ReAct prompting]
---

# ReAct

## 定义
Yao et al. (2022) 提出的 LLM prompting 框架，将推理（Reasoning）和行动（Acting）交织在一起：模型先 think（生成推理步骤），再 act（调用外部工具/API），再 observe（处理工具返回结果），循环迭代直到任务完成。

## 数学形式
$$\tau = (o_0, a_0, o_1, a_1, \ldots) \quad a_t \in \{\text{Think}(\cdot), \text{Act}(\cdot)\}$$

思考步骤不改变环境状态，行动步骤触发外部工具调用并获取 observation $o_{t+1}$。

## 核心要点
1. 交织推理和行动，克服纯推理（hallucination）和纯行动（无法复杂规划）的局限
2. 工具调用标准化：搜索、查询、执行等操作通过统一接口表达
3. 是现代 AI agent 框架（LangChain、AutoGPT 等）的理论基础
4. 在 embodied AI 中被借鉴为 VLM + 工具调用的规划范式
5. 局限：工具集需预定义，动态 tool discovery 是开放问题

## 代表工作
- [[Guava]]: 借鉴 ReAct 框架做 embodied manipulation 工具调用
- Yao et al. (2022): 原始 ReAct 论文（ALFWorld、WebShop 评测）

## 相关概念
- [[PDDL]]
- [[VLA]]
