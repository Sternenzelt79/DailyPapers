---
type: concept
aliases: [Trajectory Reachability Metric, Horizon-Matched TRM]
---

# TRM

## 定义
Trajectory Reachability Metric，用 horizon 匹配的可达性度量替换 latent MPC 中的 Euclidean terminal cost，修复 latent 空间距离不等于实际可达性的问题。

## 核心要点
1. 问题：latent MPC 用 Euclidean distance 排名候选动作序列，但 raw latent L2 不反映真实可达性
2. TRM 在训练时引入 horizon-matched 可达性监督
3. 在 PushT 和 TwoRoom 任务上验证（基于 PLDM/LeWM）

## 代表工作
- [[TRM]]: Li Liangyu et al., 2026 (arXiv 2605.22164)

## 相关概念
- [[LeWM]]
- [[PLDM]]
- [[Model Predictive Control]]
