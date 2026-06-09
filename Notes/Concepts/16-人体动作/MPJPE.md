---
type: concept
aliases: [Mean Per Joint Position Error, 均关节位置误差]
---

# MPJPE

## 定义
均关节位置误差（Mean Per Joint Position Error）：人体姿态估计的标准评估指标，计算预测关节位置与真值关节位置的平均欧氏距离（单位：毫米）。

## 数学形式
$$
\text{MPJPE} = \frac{1}{N} \sum_{i=1}^{N} \| \hat{p}_i - p_i \|_2
$$
其中 $\hat{p}_i$ 是第 $i$ 个关节的预测三维位置，$p_i$ 是对应真值，$N$ 为关节数。

## 核心要点
1. 数值越低越好，典型优秀值在人体姿态估计任务中 < 50mm
2. **PA-MPJPE**（Procrustes Aligned）：先做刚体对齐再算误差，消除全局旋转/平移影响，更关注关节相对关系
3. 在 HOI（人体-物体交互）重建中，MPJPE 衡量运动估计质量
4. [[GRAIL]] 用 MPJPE 评估其 HOI 轨迹重建的精度

## 代表工作
- [[GRAIL]]: 在 humanoid loco-manipulation 中用于评估人体运动重建

## 相关概念
- [[SMPL]]
- [[GRAIL]]
