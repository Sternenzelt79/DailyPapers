---
concept: BimanualYAM
aliases: [MolmoAct2-BimanualYAM Dataset]
category: Data
tags: [dataset, bimanual, manipulation, open-data]
created: 2026-05-09
---

# BimanualYAM Dataset

## 定义

BimanualYAM 是 [[MolmoAct2]] 项目自建的开源双臂操作数据集，截至 2026 年是**最大公开双臂数据集**。

## 规模

| 项目 | 数值 |
|------|------|
| 总时长 | 720+ 小时 |
| 演示数 | 34.5k |
| 任务数 | 28（真实世界） |
| 采集周期 | 2 个月 |
| 硬件成本 | < $6,000 USD |

## 设计要点

1. **严格失败重试协议**: 限制每条 demo 的重试次数，提升数据质量。
2. **专家遥操作**: 训练有素的操作员。
3. **YAM 双臂平台**: 低成本桌面双臂。
4. **完全开放**: 数据 + 收集协议 + 硬件清单。

## 代表工作

- [[MolmoAct2]] (2026): 提出并用于 VLA 预训练。

## 相关概念

- [[Bimanual Manipulation]]
- [[DROID]]
- [[Imitation Learning]]
