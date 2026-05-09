---
concept: MolmoAct2-Think
aliases: [Adaptive Depth Reasoning, Think Variant]
category: Architecture
tags: [vla, adaptive-computation, depth-reasoning]
created: 2026-05-09
---

# MolmoAct2-Think

## 定义

MolmoAct2-Think 是 [[MolmoAct2]] 的自适应深度推理变体，利用帧间时序冗余跳过未变化区域的深度计算。

## 核心机制

1. **场景量化**: 深度图量化为 $10 \times 10$ 网格 = 100 空间码位，128 个学习深度码值。
2. **更新 mask**:

$$
m_{t,i} = \mathbb{1}[\,\cos(x_{t,i}, x_{t-1,i}) < 0.996\,]
$$

3. **门控聚合**:

$$
c_\ell = \frac{\sum_t A_t (1-M_t) V^{vlm}_{\ell,t}}{\sum_t A_t (1-M_t)}, \qquad g_\ell = \sigma(w_\ell c_\ell + b_\ell)
$$

4. **KV 复用**:

$$
\bar{K}^{vlm}_{\ell,t} = (1 - M_t + M_t g_\ell)\, K^{vlm}_{\ell,t}
$$

## 收益

- LIBERO 平均 98.1%（vs 普通 MolmoAct2 97.2%）
- 长程任务（Pack Box / Rotate Valve）尤其受益

## 代表工作

- [[MolmoAct2]] (2026)

## 相关概念

- [[Adaptive Computation]]
- [[Cross-Attention]]
