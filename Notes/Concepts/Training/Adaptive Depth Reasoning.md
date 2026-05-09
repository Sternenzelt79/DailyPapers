---
concept: Adaptive Depth Reasoning
aliases: [Adaptive Computation, Adaptive Depth]
category: Training
tags: [adaptive-computation, efficiency, vla]
created: 2026-05-09
---

# Adaptive Depth Reasoning

## 定义

利用时序冗余动态决定哪些区域 / 层 / token 需要重新计算，跳过未变化部分以节省算力。

## 在 [[MolmoAct2-Think]] 中的实现

帧间 cosine 相似度低于阈值 0.996 才重新算深度：

$$
m_{t,i} = \mathbb{1}[\,\cos(x_{t,i}, x_{t-1,i}) < 0.996\,]
$$

未更新区域复用上一帧的 KV，更新区域用门控 $g_\ell = \sigma(w_\ell c_\ell + b_\ell)$ 缩放。

## 代表工作

- [[MolmoAct2-Think]] (2026)

## 相关概念

- [[Cross-Attention]]
- [[KV Cache]]
