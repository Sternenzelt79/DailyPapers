---
type: concept
aliases: [Dynamic Motor Control]
---

# DynaMo

## 定义
一种用于学习视觉运动表征的自监督方法，通过预测动作条件下的下一帧特征（前向预测）和从状态对反推动作（逆动力学）来训练编码器，无需外在奖励。

## 数学形式
$$\mathcal{L} = \mathcal{L}_{\text{fwd}}(\hat{z}_{t+1}, z_{t+1}) + \lambda \mathcal{L}_{\text{inv}}(\hat{a}_t, a_t)$$

## 核心要点
1. 同时优化前向预测和逆动力学两个辅助任务
2. 与 [[ICM]] 类似但在 representation learning 而非探索场景下使用
3. 作为基线出现在多个世界模型和 VLA 对比实验中

## 代表工作
- [[SMWM]]: 对比基线

## 相关概念
- [[ICM]]
- [[JEPA]]
- [[SMWM]]
