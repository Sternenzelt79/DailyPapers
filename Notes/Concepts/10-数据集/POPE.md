---
type: concept
aliases: [POPE, Polling-based Object Probing Evaluation]
---

# POPE

## 定义
用于评估多模态大语言模型（MLLM）幻觉（hallucination）问题的 benchmark，通过是否存在问题（"图中有 XX 吗？"）量化模型对不存在物体的错误感知率。

## 核心要点
1. 三种采样策略：random、popular、adversarial（难度递增）
2. 指标：Accuracy、Precision、Recall、F1
3. 反映 MLLM 对物体存在性的判断是否可靠
4. [[Eagle]]、[[InternVL]] 等 MLLM 均在此 benchmark 上报告结果

## 代表工作
- Li et al. (2023), "Evaluating Object Hallucination in Large Vision-Language Models"

## 相关概念
- [[MME]]
- [[InternVL]]
- [[CLIP]]
