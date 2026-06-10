---
type: concept
aliases: [SHAC, Short-Horizon Actor-Critic]
---

# SHAC

## 定义
SHAC（Short-Horizon Actor-Critic）是一种利用可微仿真梯度的策略优化方法，通过短时间窗口截断反传来平衡梯度质量与计算效率，适用于高维连续控制任务。

## 数学形式
$$\nabla_\theta J \approx \mathbb{E}\left[\sum_{t=0}^{H}\gamma^t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q_t\right]$$

## 核心要点
1. 利用可微仿真（IsaacGym 等）直接反传动力学梯度
2. 短时间窗口 H 避免梯度爆炸
3. 结合 Actor-Critic 减少方差

## 代表工作
- Geilinger et al. (2023), SHAC: Short-Horizon Actor-Critic

## 相关概念
- [[MAD]]
- [[PPO]]
