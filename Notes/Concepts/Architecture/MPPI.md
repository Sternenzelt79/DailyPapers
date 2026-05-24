---
type: concept
aliases: [Model Predictive Path Integral, MPPI, 路径积分控制]
---

# MPPI（Model Predictive Path Integral）

## 定义
一种基于重要性采样的无梯度轨迹优化算法，通过并行采样大量扰动轨迹并以指数加权平均更新控制序列，适用于非线性、不可微系统的实时 MPC 规划。

## 数学形式

$$
\mathbf{u}^* = \mathbf{u} + \frac{\sum_{k=1}^{K} w_k \cdot \epsilon_k}{\sum_{k=1}^{K} w_k}
$$

其中权重：

$$
w_k = \exp\!\Big(-\frac{1}{\lambda} S_k\Big), \quad S_k = \sum_t c(\hat{s}_t^{(k)}, u_t + \epsilon_t^{(k)})
$$

## 核心要点
1. 无需代价函数可微，适用于接触丰富的机器人任务
2. 通过温度参数 $\lambda$ 控制探索-利用权衡
3. 天然并行化，适合 GPU 加速
4. 比 CEM 对噪声更鲁棒，因为使用软加权而非硬截断

## 代表工作
- [[stable-worldmodel]]: 作为采样类规划求解器之一（2026）

## 相关概念
- [[CEM]]
- [[模型预测控制]]
- [[世界模型]]
