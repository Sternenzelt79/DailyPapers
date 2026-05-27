---
type: concept
aliases: [FoundationPose]
---

# FoundationPose

## 定义
NVIDIA 提出的通用 6-DoF 物体位姿估计与追踪框架，支持有/无 CAD 模型两种模式，是目前机器人操作中最广泛使用的基础位姿估计工具之一。

## 数学形式
$$\hat{T}_{obj} = \arg\max_T \log p(I \mid T, M)$$

给定 RGB-D 图像 $I$ 和物体模型 $M$（CAD 或 few-shot 参考图），估计物体在相机坐标系下的变换 $T \in SE(3)$。

## 核心要点
1. 支持 model-based 和 model-free 两种模式（有/无 CAD 模型）
2. 渲染-比较框架：生成候选位姿渲染，与观测图像对比评分
3. 泛化能力强，可零样本应用于新物体
4. 在机器人视频世界模型中用于提供 4D 对应的位姿锚点

## 代表工作
- [[GEM-4D]]: 用 FoundationPose 提供位姿估计，与 CoTracker3 联合构建 4D 对应监督

## 相关概念
- [[CoTracker3]]
- [[VGGT]]
