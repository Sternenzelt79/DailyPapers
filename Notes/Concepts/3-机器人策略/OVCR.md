---
type: concept
aliases: [OVCR, Observation-guided Context Routing]
---

# OVCR

## 定义
AHA-WAM 提出的观测引导上下文路由机制，根据当前观测动态决定哪些历史视觉上下文被激活并传入世界模型，实现视觉预测与动作执行的异步解耦。

## 核心要点
1. 视觉变化慢（低频更新），动作需要高频响应，OVCR 解耦两者时间分辨率
2. 根据观测相似度决定是否触发新的世界模型预测，相似时复用缓存上下文
3. 与 [[C3ache]] 的关键区别：C3ache 是跨推理步骤的 chunk 特征缓存，OVCR 是上下文路由决策

## 代表工作
- [[AHA-WAM]]: 提出 OVCR，用于异步时序世界-动作建模

## 相关概念
- [[WAM]]
- [[World Action Model]]
