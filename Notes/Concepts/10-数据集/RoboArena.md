---
type: concept
aliases: [RoboArena, Robo Arena]
category: Datasets
tags: [benchmark, policy-evaluation, robot-manipulation, droid]
created: 2026-06-04
---

# RoboArena

用于机器人策略比较评估的 benchmark，基于 DROID 平台（Franka Panda），通过 Bradley-Terry 模型对多个策略进行排名评估。

## 核心特点

- **平台**: Franka Panda（DROID 机器人）
- **规模**: 65 个评估 session，7 个不同 DROID-based 策略
- **任务**: 包括"把食物放到盘子上"、"按手机按钮"、"把面包移到盘子上"等
- **评分机制**: [[Bradley-Terry]] 模型，通过成对比较计算策略 Elo 排名

## 用途

可用作世界模型策略评估代理的验证集：在 OSCAR 生成视频上的虚拟评估结果与 RoboArena 真实评估结果 Spearman 相关系数达 0.750。

## 代表工作

- [[OSCAR]]: 使用 RoboArena 验证虚拟策略评估的可靠性

## 相关概念

- [[DROID]]: 数据和平台来源
- [[Bradley-Terry]]: 排名统计模型
