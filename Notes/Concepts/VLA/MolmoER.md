---
concept: MolmoER
aliases: [Molmo2-ER, Molmo Embodied Reasoning]
category: VLA
tags: [vlm, embodied-ai, foundation-model]
created: 2026-05-09
---

# MolmoER (Molmo2-ER)

## 定义

MolmoER 是 [[MolmoAct2]] 的具身推理 VLM 主干（4B），由通用 [[Molmo]] 经 specialize-then-rehearse 训练得到，强化具身 QA、Pointing、Ego-Exo 多视角对齐能力。

## 训练数据 (3.26M 具身 + 9.25M 通用)

- Image Embodied QA: 1.33M
- Image Pointing: 780K
- Image Detection: 100K
- Video Embodied QA: 703K
- Multi-image / Ego-Exo: 700K
- Abstract Reasoning: 150K

## 核心要点

1. **Specialize-then-Rehearse**: 先在具身专用数据上专门化，再混入通用 Molmo 语料防遗忘。
2. **基准成绩**: 13 项具身推理基准平均 63.8%，超越 GPT-5 (57.9%) 与 GR-ER 1.5 Thinking (61.3%)。
3. **小而强**: 4B 参数，相比 baseline Molmo2 提升 17 pp。

## 代表工作

- [[MolmoAct2]] (2026): MolmoER 作为 VLA 的视觉-语言主干。

## 相关概念

- [[Vision-Language-Action Model]]
- [[Embodied QA]]
- [[Pointing]]
