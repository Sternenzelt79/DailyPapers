---
type: concept
aliases: [LeWorldModel, Le World Model, LeWM]
---

# LeWM（Le World Model）

## 定义
基于 LeRobot 生态系统的潜在世界模型，是 PLDM 的简化版本，将原 PLDM 的约 7 项 loss 精简为 2 项 loss，在降低工程复杂度的同时超越原版性能。

## 核心要点
1. 简化了 PLDM 的多项式正则化，仅保留 2 项核心 loss
2. 在 Push-T 任务达到 94% 成功率（In-Distribution），超越 PLDM（78%）和 DINO-WM（92%）
3. 在 OGBench-Cube 上为 72%，低于 DINO-WM（86%），说明不同任务性能存在差异
4. 是 stable-worldmodel 平台收录的四个潜在世界模型基线之一

## 代表工作
- [[stable-worldmodel]]: 作为世界模型基线，Push-T 任务最优（2026）

## 相关概念
- [[PLDM]]
- [[DINO-WM]]
- [[潜在世界模型]]
- [[World Model]]
