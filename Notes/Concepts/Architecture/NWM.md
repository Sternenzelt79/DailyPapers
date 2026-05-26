---
type: concept
aliases: [NWM, Navigation World Model]
---

# NWM（Navigation World Model）

## 定义
基于条件扩散 Transformer 的导航世界模型：输入历史 ego-view 图像序列和动作，生成未来观测帧，支持基于 rollout 的导航规划（CEM/MPC）。

## 数学形式
$$o_{t+1} \sim p_\theta(o_{t+1} | o_{\leq t}, a_{\leq t})$$

递归地以历史观测和动作为条件，生成下一帧 ego-view 图像。

## 核心要点
1. **Conditional DiT 骨干**：基于 [[DiT]] 架构的条件视频扩散模型
2. **Rollout 规划**：生成未来多步预测用于 CEM/MPC 规划，选择最优动作序列
3. **漂移问题**：自回归 rollout 存在感知漂移（噪声累积）和几何漂移，[[DR-NWM]] 针对此做了改进

## 代表工作
- Bar et al.《Navigation World Models》
- [[DR-NWM]]：在 NWM 基础上加 epipolar anchor 解决双类漂移

## 相关概念
- [[World Model]]
- [[DiT]]
- [[DR-NWM]]
- [[Model Predictive Control]]
