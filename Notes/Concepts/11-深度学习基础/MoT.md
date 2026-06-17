---
type: concept
aliases: [Mixture of Transformers, MoT]
---

# MoT

## 定义
Mixture of Transformers：将不同模态或任务的 token 路由到专门的 transformer expert，共享基础 backbone 的同时避免表示空间冲突。

## 数学形式
$$y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)$$

其中 $g_i(x)$ 为路由权重，$E_i$ 为第 $i$ 个 expert transformer block。

## 核心要点
1. 不同于 MoE（专家在 FFN 层），MoT 在整个 transformer block 层面路由
2. 适用于多模态统一模型：理解 token 和生成 token 走不同路径
3. 节省参数的同时保持各模态专属能力

## 代表工作
- [[SenseNova-U1]]: NEO-unify 架构使用 MoT 统一理解与生成
- [[Cosmos3]]: omnimodal world model 也采用 MoT 结构

## 相关概念
- [[OmniGen2]]
- [[BAGEL]]
