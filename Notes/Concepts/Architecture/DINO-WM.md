---
type: concept
aliases: [DINO World Model]
---

# DINO-WM

## 定义
基于冻结的 DINO 自监督特征训练 latent 动力学预测器与规划器的 world model。

## 核心要点
1. Encoder 来自预训练 DINO，不参与训练。
2. 规划开销大（depending on backbone）。
3. 是 [[LeWM]] 的主要对比基线之一，被后者在速度上拉开 ~48×。

## 代表工作
- DINO-WM (Zhou et al. 2024)
- [[stable-worldmodel]]: 评测中 OGBench-Cube 最优（86%），Push-T 次优（92%）（2026）

## 相关概念
- [[World Model]]
- [[JEPA]]
- [[LeWM]]
