---
concept: Adaptive Depth Reasoning
category: Architecture
tags: [vla, world-model, depth, efficient-inference, reasoning]
created: 2026-05-09
---

# Adaptive Depth Reasoning

## 定义

一种把"思考"代价随时间局部性按需触发的机制：把场景深度编码为离散 token 网格，每个时间步只对**RGB 发生变化**的 patch 重新预测对应的 [[Depth Token|深度 token]]，未变化区域沿用上一步缓存。深度 token 作为隐式 [[World Model|世界模型]] 注入动作专家。

## 数学形式

变化 mask 与缓存更新规则：

$$
\begin{aligned}
m_{t,i} &= \mathbf{1}\!\left[\cos(x_{t,i}, x_{t-1,i}) < 0.996\right] \\
b_{t,i} &= \begin{cases} d_{t,i} & m_{t,i} = 1 \\ b_{t-1,i} & m_{t,i} = 0 \end{cases}
\end{aligned}
$$

其中 $x_{t,i}$ 是第 $i$ 个 32×32 RGB patch 的特征，$d_{t,i}$ 是新预测的深度 code。

## 核心要点

1. **10×10 深度网格**：每帧 100 个深度 token，从 128 大小的 [[VQ Codebook|VQ 码本]] 取出。
2. **patch 级时间局部性**：cos 相似度 < 0.996 才触发重预测。
3. **门控注入**：未变化区域通过 sigmoid gate $g_\ell$ 弱化对动作专家的贡献。
4. **三种训练输出风格**：depth-only / action-only / depth-and-action 等概率采样。
5. 推理时只自回归解码变化 cell，平均代价显著低于完整 [[Chain-of-Thought|思考链]]。

## 代表工作

- [[MolmoAct2]] / MolmoAct2-Think: 首次在 VLA 中引入此机制，LIBERO 平均 98.1%，长程任务 +3 点。

## 相关概念

- [[World Model]]
- [[Depth Token]]
- [[VQ Codebook]]
- [[Cosine Similarity]]
- [[Chain-of-Thought]]
