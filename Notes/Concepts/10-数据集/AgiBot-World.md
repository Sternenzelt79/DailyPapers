---
type: concept
aliases: [AgiBot World, AgiBot数据集]
---

# AgiBot-World

## 定义

AgiBot-World 是由 AgiBot 发布的大规模多视图机器人操作视频数据集，包含多摄像头视角的操作演示，同时也被用作世界模型生成质量的评估 benchmark。

## 核心要点

1. **多视图**: 覆盖多个固定摄像头角度，适合训练和评估 3D 一致性
2. **大规模**: 提供大量多样化操作任务视频片段
3. **双重用途**: 既作为训练数据集，也作为文本/动作条件生成的评估平台
4. **衍生竞赛**: AgiBot-Challenge2026 基于此数据集评估世界模型

## 评估指标

- SSIM（结构相似性）
- LPIPS（感知图像质量）
- FID（Fréchet 图像距离）
- FVD（Fréchet 视频距离）
- Scene Consistency（场景一致性）
- MEt3R（多视图 3D 重建误差）

## 代表工作

- [[PAIWorld]]: 使用 AgiBot-World 35% 数据训练，并在其上评估文本条件多视图生成

## 相关概念

- [[WorldArena]]
- [[RoboMIND]]
