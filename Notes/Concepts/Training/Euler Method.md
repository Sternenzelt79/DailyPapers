---
type: concept
category: Training
tags: [ode, sampling]
created: 2026-05-09
---

# Euler Method

最简单的 ODE 一阶积分：$x_{i+1} = x_i + (\tau_{i+1}-\tau_i) f(x_i, \tau_i)$。在 [[Flow Matching]] 推理中常用，几步即可得到不错的样本。

## 代表工作

- [[RLDX-1]]: 推理时用 Euler 法对学好的速度场积分，少量步数生成动作块。
