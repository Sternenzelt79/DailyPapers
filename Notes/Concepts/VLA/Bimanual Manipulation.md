---
concept: Bimanual Manipulation
aliases: [双臂操作, Two-Arm Manipulation]
category: VLA
tags: [manipulation, bimanual, robotics]
created: 2026-05-09
---

# Bimanual Manipulation

## 定义

双臂操作指机器人用两只手协同完成任务（穿绳、拆装、传递、双手抓取）。对动作维度、协调、时序对齐都有更高要求。

## 难点

1. **高维动作空间**: 动作维度翻倍，搜索空间指数增长。
2. **臂间协调**: 时序、力闭环、避碰。
3. **数据稀缺**: 公开数据集体量远小于单臂 [[DROID]]。

## 关键数据集

- [[ALOHA]]: 早期低成本双臂遥操作。
- [[BimanualYAM]] (2026, [[MolmoAct2]]): 720h，目前最大开放双臂数据集。

## 代表工作

- [[MolmoAct2]]
- [[ALOHA]]
- [[Mobile ALOHA]]

## 相关概念

- [[Action Chunking]]
- [[Imitation Learning]]
