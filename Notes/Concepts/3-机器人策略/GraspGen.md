---
type: concept
aliases: [GraspGen, 抓取生成]
---

# GraspGen

## 定义
基于扩散的 6-DOF 抓取生成框架，在点云上预测抓取候选集，支持"on-generator training"以提升抓取质量和泛化能力。

## 核心要点
1. 以目标物体点云为输入，生成多个 6-DoF 抓取候选姿态 $\{(\mathbf{R}^i_{\text{grasp}}, \mathbf{T}^i_{\text{grasp}})\}$
2. 采用扩散框架建模抓取分布，支持多样化抓取候选采样
3. 支持 on-generator training，即在生成器自身输出上迭代训练以提升质量

## 代表工作
- [[GEM-4D]]: 在 Adaptive Inverse Dynamic System 中使用 GraspGen 预测抓取候选，并按与参考姿态的加权位姿偏差排序选择最优抓取

## 相关概念
- [[FoundationPose]]
- [[帧间对应]]
- [[3-机器人策略]]
