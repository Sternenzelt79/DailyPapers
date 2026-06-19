---
type: concept
aliases: [Latent-Aware Controller, 潜变量感知控制器]
---

# LAC（Latent-Aware Controller）

## 定义

一种基于潜变量语义的部署时自适应动作平滑控制器，通过分析共享能量、残差能量和残差对立信号，在宏观运动、精细调整和噪声抑制三种执行模式间动态切换，无需专用力/阻抗控制硬件。

## 数学形式

$$
\tilde{q}_t = (1 - \alpha_t) \cdot \tilde{q}_{t-1} + \alpha_t \cdot q_t
$$

$$
\rho_t = \frac{E^r_t}{E^s_t + \varepsilon}, \quad \omega_t = -\frac{\langle a^r_{t,L}, a^r_{t,R} \rangle}{\|a^r_{t,L}\|_2 \cdot \|a^r_{t,R}\|_2 + \varepsilon}
$$

## 核心要点

1. **微运动比** $\rho_t$：残差能量相对共享能量的比例，高值指示需要精细控制
2. **对立信号** $\omega_t$：双臂残差方向的对立程度，高值指示协同夹持/稳定行为
3. **三种模式**：宏观主导（大 $\alpha$，高响应）/ 精细调整（中 $\alpha$，保留微运动）/ 噪声抑制（小 $\alpha$，平滑）
4. 时间平滑系数 $\beta=0.2$ 防止模式切换过于激烈

## 代表工作

- [[Co-VLA]]: LAC 的提出工作，在真实双臂机器人实验中验证其有效性

## 相关概念

- [[SAE|Structured Action Expert]]
- [[自适应平滑]]
- [[共享-残差分解]]
