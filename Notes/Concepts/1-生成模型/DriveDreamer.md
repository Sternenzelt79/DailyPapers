---
type: concept
aliases: [DriveDreamer, 驾驶场景世界模型]
---

# DriveDreamer

## 定义
DriveDreamer 是一种可控的驾驶场景视频生成世界模型，以结构化交通条件（HD 地图、框、轨迹）作为条件生成真实的驾驶视频。

## 核心要点
1. 用 HD 地图 + 3D bounding box + 轨迹作为多模态条件
2. 基于扩散模型，生成未来帧预测
3. 可作为数据增强或 policy 评估仿真器

## 代表工作
- Wang et al. (2023), DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving

## 相关概念
- [[UniSim]]
- [[CityDreamer]]
