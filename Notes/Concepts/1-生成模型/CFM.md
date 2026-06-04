---
type: concept
aliases: [Conditional Flow Matching, 条件流匹配]
---

# CFM

## 定义
CFM（Conditional Flow Matching）是 Flow Matching 的条件版本，在给定条件 $c$ 下学习从噪声分布到目标分布的概率流 ODE。

## 数学形式
$$\frac{dx}{dt} = v_\theta(x, t, c)$$

## 核心要点
1. 通过最小化向量场回归损失训练：$\mathcal{L} = \mathbb{E}[\|v_\theta(x_t, t, c) - u_t(x_t|x_1)\|^2]$
2. 训练稳定，不需要 DDPM 那样的 noise schedule 设计
3. 推理时用 ODE solver 从 $x_0 \sim \mathcal{N}(0,I)$ 到 $x_1$

## 代表工作
- Lipman et al. (2022), Flow Matching for Generative Modeling
- [[ForesightFlow]]: 将 potential 函数整合进 CFM 的 ODE

## 相关概念
- [[Flow Matching]]
- [[DDPM]]
