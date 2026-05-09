---
concept: Space-Time Self-Similarity
aliases: [STSS]
category: Architecture
tags: [motion, vision, vla]
created: 2026-05-09
---

# Space-Time Self-Similarity (STSS)

## 定义

STSS 是一种显式向视觉编码器注入**像素/patch 级运动感知**的模块。对一段连续视频帧，计算同一空间位置 patch 在相邻帧之间的 token 相似度矩阵，把得到的 self-similarity tensor 通过残差注入到视觉编码器中间层。

## 动机

预训练视觉模型（DINO / SigLIP / CLIP）单帧训练，缺少时序运动表征。直接堆叠帧让模型自己学运动效率低，而显式 self-similarity 能在不破坏预训练权重的前提下提供运动先验。

## 实现要点

- 注入位置：视觉编码器中段（[[RLDX-1]] 中为第 9 层）。
- 残差形式：`x = x + STSS(x)`，保留预训练特征。
- 计算成本低：仅相邻帧 patch 级 cosine。

## 代表工作

- [[RLDX-1]]: 在 conveyor belt 等动态任务上从 ~30% 提升到 87.5%。

## 相关概念

- [[Motion Awareness]]
- [[Vision Encoder]]
