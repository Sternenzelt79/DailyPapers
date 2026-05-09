---
type: concept
aliases: [V-JEPA, Video-JEPA, V-JEPA 2]
---

# V-JEPA

## 定义
视频版 [[JEPA]]：用一段视频上下文 patch 预测被 mask 的目标 patch 在 latent 中的表示，做无监督视频表示学习。Bardes et al., 2024。V-JEPA 2 (2025) 进一步扩展到 action-conditioned 视频。

## 核心要点
1. 同样不重建像素；目标是学到"可预测物理"的表征。
2. 在 action recognition / Something-Something V2 上 competitive。
3. V-JEPA 2 引入 action 条件，开始向 world model 靠拢。

## 代表工作
- V-JEPA (Bardes 2024)
- V-JEPA 2 (2025)
- [[LeWM]]：相当于 V-JEPA 的"端到端控制版"

## 相关概念
- [[JEPA]]
- [[I-JEPA]]
- [[World Model]]
