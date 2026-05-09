---
type: concept
aliases: [Diffusion Model, DDPM, Score-Based Model]
---

# Diffusion Model

## 定义
通过逐步加噪（前向）与去噪（反向）建模数据分布的生成模型；可由 score matching / DDPM / SDE 三视角统一推导。

## 数学形式
反向 SDE：
$$
d x_t = \big[f(x_t, t) - g(t)^2 \nabla_x \log p_t(x_t)\big]\,dt + g(t)\,d\bar w_t
$$

## 核心要点
1. 训练稳定、模式覆盖好、易接条件（class、text、action）。
2. 在机器人中作为 [[Diffusion Policy]] 直接输出动作分布。
3. 与 [[Flow Matching]] 紧密相关。

## 代表工作
- DDPM (Ho 2020), DDIM, EDM
- [[Diffusion Policy]] (Chi 2023)

## 相关概念
- [[Flow Matching]]
- [[Score Matching]]
- [[Diffusion Policy]]
