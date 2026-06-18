---
type: concept
aliases: [InternVLA, InternVL-based VLA]
---

# InternVLA

## 定义
基于 InternVL 系列视觉-语言模型（上海 AI Lab）fine-tuned 的 Vision-Language-Action 模型，在 VLM 预训练知识基础上通过机器人操作数据适配，用于端到端机器人策略学习。

## 数学形式
$$\pi_\theta(a_t | o_t, l) = \text{VLA-Head}(\text{InternVL}(o_t, l))$$

其中 $o_t$ 为视觉观测，$l$ 为语言指令，action head 在 InternVL encoder 输出上预测动作。

## 核心要点
1. 基于 InternVL3 系列（强大的多模态理解能力）fine-tune 为 VLA
2. 中文学术界代表性 VLA 工作之一
3. 在 LIBERO 等 benchmark 上有竞争力的性能
4. VLA fine-tuning 会导致 InternVL 预训练知识遗忘（见 VLA-KnowledgeBench）
5. 与 OpenVLA、SmolVLA、SpatialVLA 是同类竞争工作

## 代表工作
- [[VLA-KnowledgeBench]]: 测量 InternVLA fine-tuning 后知识遗忘情况
- [[InternVL3]]: 基础 VLM 模型

## 相关概念
- [[VLA]]
- [[OpenVLA]]
- [[SmolVLA]]
