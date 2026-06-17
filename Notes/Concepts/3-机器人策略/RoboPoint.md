---
type: concept
aliases: [RoboPoint, Robot Pointing]
---

# RoboPoint

## 定义
将视觉语言模型的 pointing/grounding 能力迁移到机器人操控场景，通过预测接触点（keypoint）辅助抓取规划。

## 核心要点
1. 基于 VLM 预测操控关键点坐标
2. 接触点预测用于抓取规划和末端执行器控制
3. 与 SAM、VoxPoser 等工具配合使用

## 代表工作
- [[GeneralVLA-2]]: 使用 RoboPoint 做接触点预测

## 相关概念
- [[VoxPoser]]
- [[SAM]]
