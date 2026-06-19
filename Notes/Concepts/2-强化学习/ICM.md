---
type: concept
aliases: [Intrinsic Curiosity Module, 内在好奇心模块]
---

# ICM

## 定义
Intrinsic Curiosity Module，一种基于预测误差的内在奖励机制，用前向模型在特征空间中的预测误差衡量状态的"新颖性"，驱动探索。

## 数学形式
$$r_i^t = \frac{\eta}{2} \|\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\|_2^2$$

其中 $\phi$ 为特征编码器，$\hat{\phi}$ 为前向模型预测的下一状态特征，$\eta$ 为缩放系数。

## 核心要点
1. 包含逆动力学模型（从状态对预测动作）和前向动力学模型
2. 在特征空间而非像素空间计算预测误差，减少无关噪声影响
3. 与稀疏外在奖励结合使用效果最好

## 代表工作
- Pathak et al. (2017): 原始 ICM 论文，在 Atari 和 VizDoom 上验证
- [[SMWM]]: 使用逆动力学正则化与 ICM 思路相关

## 相关概念
- [[JEPA]]
- [[DynaMo]]
