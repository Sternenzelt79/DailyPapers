---
type: concept
aliases: [FastV, Fast Visual Token Pruning]
---

# FastV

## 定义
FastV 是一种面向视觉语言模型（VLM）的 visual token 剪枝方法，通过在 Transformer 的早期层中基于 attention score 识别并丢弃低重要性的视觉 token，在推理阶段减少计算开销。

## 核心要点
1. 在 VLM 的中间层（而非输入端）执行 token 剪枝，保留语义重要的视觉 token
2. 通过 cross-attention score 排序确定 token 重要性
3. 以较低的精度损失换取显著的推理加速（通常 30-50% FLOPs 降低）
4. 直接用于 VLA 时存在 "semantic-action gap"：对 VLM 有用的 token 不等于对 action 预测有用的 token

## 局限性
- 语义重要性排名 ≠ 动作预测重要性（VLA-Pruner 针对此问题提出改进）
- 剪枝比例对性能影响敏感，需要任务级调优

## 代表工作
- FastV (Kong et al.): 原始方法，用于 VLM 加速
- [[VLA-Pruner]]: 指出 FastV 在 VLA 上的局限并提出改进

## 相关概念
- [[SparseVLM]]
- [[EfficientVLA]]
- [[Vision Transformer]]
