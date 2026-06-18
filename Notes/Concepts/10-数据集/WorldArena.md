---
type: concept
aliases: [WorldArena Benchmark, WorldArena排行榜]
---

# WorldArena

## 定义

WorldArena 是一个评估机器人操作世界模型（World Foundation Models）**动作条件生成**质量的综合 benchmark，设有公开排行榜，以 EWMScore 作为核心综合指标。

## 核心要点

1. **综合评估**: EWMScore 综合以下 6 个子维度：
   - Visual Quality（视觉质量）
   - Motion Quality（运动质量）
   - Content Consistency（内容一致性）
   - Physics Adherence（物理合理性）
   - 3D Accuracy（3D 精度）
   - Controllability（可控性）
2. **动作条件生成**: 给定历史帧和动作序列，评估未来帧预测质量
3. **竞争排行榜**: 参与方包括 MAI、Pelican-Unify、BWM-Fast、SparkWorld 等多支团队

## 当前排名（截至 2026-06）

| 排名 | 方法 | EWMScore |
|------|------|----------|
| 1 | PAIWorld | 70.67 |
| 2 | GenieEnvisioner-Sim2.0-2B | 68.26 |
| 3 | BWM-Fast | 67.87 |

## 代表工作

- [[PAIWorld]]: WorldArena 第一名（EWMScore 70.67，Motion Quality 79.66 全场最优）

## 相关概念

- [[AgiBot-World]]
