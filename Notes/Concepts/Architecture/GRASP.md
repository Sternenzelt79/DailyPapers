---
type: concept
aliases: [GRASP, Gradient-Assisted Sampling Planning]
---

# GRASP（Gradient-Assisted Sampling Planning）

## 定义
一种结合梯度信息和采样的规划求解器，利用世界模型的可微性提供梯度引导，再通过采样扩展覆盖非凸区域，兼顾梯度方法的精度和采样方法的鲁棒性。

## 核心要点
1. 在梯度类规划求解器中引入采样扰动，避免局部最优
2. 需要世界模型可微（不适用于不可微模拟器）
3. 是 stable-worldmodel 平台六种规划求解器之一
4. 相比纯梯度方法，对局部最优更鲁棒

## 代表工作
- [[stable-worldmodel]]: 作为梯度辅助采样规划求解器（2026）

## 相关概念
- [[CEM]]
- [[MPPI]]
- [[模型预测控制]]
