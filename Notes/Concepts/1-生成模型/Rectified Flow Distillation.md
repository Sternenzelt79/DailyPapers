---
type: concept
aliases: [Rectified Flow Distillation, 整流流蒸馏]
---

# Rectified Flow Distillation

## 定义
基于 Rectified Flow 的知识蒸馏技术，通过辅助训练步骤将多步采样的流模型压缩为少步（甚至单步）高质量生成模型，同时保持生成质量。

## 数学形式
蒸馏目标：最小化学生模型（少步）与教师模型（多步）输出分布的差异：
$$
\mathcal{L}_{\text{distill}} = \mathbb{E}\left[\| x_{\text{student}} - x_{\text{teacher}} \|^2_2\right]
$$

学生模型使用更大步长或更少 NFE（Number of Function Evaluations）完成采样。

## 核心要点
1. **减少 NFE**: 将原本需要 50 步以上的采样压缩到 4-8 步
2. **后训练方案**: 不改变模型结构，只需额外的蒸馏微调步骤
3. **余弦调度**: 常与渐进式余弦噪声调度配合使用，在少步内最大化信息量

## 代表工作
- [[WEAVER]]: 通过蒸馏后训练将推理步数大幅减少，实现 5-10× 总体推理加速

## 相关概念
- [[Rectified Flow]]
- [[Flow Matching]]
- [[Diffusion Forcing]]
