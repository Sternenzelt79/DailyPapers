---
type: concept
aliases: [Imagine, Refine, Iterate Strategies, IRIS World Model]
---

# IRIS

## 定义
基于 Transformer 的离散世界模型，通过 VQ-VAE 将环境编码为离散 token，在想象的潜空间中进行 RL 规划，在 Atari 游戏上取得优秀性能。

## 数学形式
$$z_t = \text{VQ-VAE}(o_t), \quad \hat{z}_{t+1} = \text{Transformer}(z_{1:t}, a_{1:t})$$

## 核心要点
1. VQ-VAE 离散化观测为 token 序列
2. Transformer 在 token 空间预测未来状态
3. 在想象轨迹上用 RL 训练策略
4. 与 DreamerV3 并列为代表性基于模型的 RL 方法

## 代表工作
- [[LoopWM]]: 以 IRIS 为对比基线

## 相关概念
- [[DreamerV3]]
- [[MuZero]]
- [[World Model]]
