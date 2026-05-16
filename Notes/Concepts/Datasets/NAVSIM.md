---
type: concept
aliases: [NAVSIM, Non-Reactive Autonomous Vehicle Simulation]
---

# NAVSIM

## 定义
NAVSIM 是一个非反应式自动驾驶规划 benchmark，使用真实驾驶数据回放而非交互式仿真，评估轨迹预测模型在真实场景下的规划质量。

## 数学形式
$$\text{PDMS} = \prod_{k} w_k \cdot m_k$$

其中 $m_k$ 是各子指标（碰撞率、舒适度、进度等），$w_k$ 为对应权重。

## 核心要点
1. 使用 nuPlan 数据集的真实驾驶片段，不运行实时仿真引擎
2. 评估指标为 PDMS（Planning-oriented Driving Score），覆盖安全性、舒适性、行进进度
3. 非反应式设计减少了仿真偏差，但也意味着无法测试因果交互

## 代表工作
- [[OneVL]]: 在 NAVSIM 上评估隐式 CoT 规划方法

## 相关概念
- [[VLA]]（视觉-语言-动作模型，常用于自动驾驶规划）
- [[World Model]]（用于自动驾驶的世界建模）
