---
type: concept
aliases: [Navigation World Model, NWM, 导航世界模型]
---

# Navigation World Model

## 定义
一类视觉导航模型，显式建模 $p_\theta(o_{t+H} \mid o_t, a_{t:t+H-1})$，即给定当前观测和候选动作序列预测未来自我中心观测，通过对大量候选动作的预测结果评分来选择最优动作。

## 数学形式

$$
p_\theta(o_{t+H} \mid o_t, a_{t:t+H-1})
$$

## 核心要点
1. 将视觉预见（visual foresight）显式纳入导航决策，增强对复杂场景的长程感知
2. 推理时需要外部规划器（如 [[CEM]]）对多个候选动作分别调用模型，计算开销随候选数线性增长
3. 外部规划依赖是主要瓶颈：闭环行为取决于规划器配置而非模型本身
4. 代表性工作包括 PathDreamer、Schrödinger's Navigator 等

## 代表工作
- [[NavWAM]]: 将 NWM 与直接策略统一，消除外部规划器依赖

## 相关概念
- [[World-Action Model]]
- [[CEM]]
- [[ViNT]]
- [[GNM]]
