---
type: concept
aliases: [AnyGrasp, 通用抓取]
---

# AnyGrasp

## 定义
AnyGrasp：通用抓取检测方法，直接从点云/RGB-D 输入中预测 6-DoF 抓取姿态，支持任意形状物体的零样本抓取，是机器人操作中最常用的抓取检测 baseline 之一。

## 数学形式
$$\mathcal{G} = f_\theta(\mathcal{P}), \quad \mathcal{G} = \{(R_i, t_i, w_i, q_i)\}$$

## 核心要点
1. 输入点云，输出候选抓取姿态集合（位置 + 方向 + 宽度 + 置信度）
2. 基于 PointNet++ 或类似点云 encoder，直接回归抓取参数
3. 大规模数据集训练（GraspNet-1Billion），支持 open-set 物体
4. 常作为机器人操作实验的抓取 baseline 或预训练模块

## 代表工作
- AnyGrasp 论文（Fang et al.，具体 arXiv 待查）
- GraspNet-1Billion 数据集配套方法

## 相关概念
- [[VLA]]
- [[Diffusion Policy]]
