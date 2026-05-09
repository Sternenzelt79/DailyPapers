---
concept: Molmo2-ER
category: Architecture
tags: [vlm, embodied-reasoning, backbone]
created: 2026-05-09
---

# Molmo2-ER

## 定义

由 [[Molmo2]] 经"specialize-then-rehearse"两阶段训练得到的 4B 参数具身推理专用 VLM。在 13 个具身/空间推理 benchmark 上平均 63.8%，超过 GPT-5（57.9%）与 Gemini 2.5 Pro（57.1%）。

## 核心要点

1. **Stage 1（具身专门化）**：20K 步，在 3.3M 样本（Embodied QA / Pointing / Tracking）+ 8% Tulu-3 文本上训练，序列 4,200，batch 64 × 8×H100。
2. **Stage 2（联合精修）**：1.5K 步，混入原始 Molmo2 多模态语料（具身/通用 50/50），序列扩到 16,384，防止灾难性遗忘。
3. **作为 [[VLA]] 主干**：在 [[MolmoAct2]] 中提供 KV 缓存供流匹配动作专家通过 [[KV-Cache Conditioning|KV-Cache 条件化]] 接入。

## 评测 benchmark

Point-Bench、RefSpatial、RoboSpatial-Poi、Where2Place、BLINK、CV-Bench、ERQA、EmbSpatial、MindCube、RoboSpatial-VQ、SAT、OpenEQA、VSI-Bench

## 代表工作

- [[MolmoAct2]]: 作为 VLM 主干使用。

## 相关概念

- [[Molmo2]]
- [[Embodied QA]]
- [[Image Pointing]]
- [[Tulu-3]]
