---
type: concept
aliases: [Navigation Diffusion Policy]
---

# NavDP

## 定义
基于扩散策略（Diffusion Policy）的视觉导航方法，将导航轨迹生成建模为扩散过程，支持多目标（image-goal、point-goal）导航。

## 核心要点
1. 将 [[扩散策略]] 的框架迁移到导航任务，生成连续平滑的轨迹
2. 条件可以是目标图像（image-goal）或目标坐标（point-goal）
3. 被 WAM-Nav 作为 baseline 对比

## 代表工作
- [[WAM-Nav]]: 与 NavDP 对比，用 asymmetric WM 做导航

## 相关概念
- [[扩散策略]] — 基础方法
- [[WAM]] — 世界动作模型框架
- [[NWM]] — 神经世界模型用于导航
