---
type: concept
aliases: [解耦噪声调度, Asymmetric Denoising, 非对称去噪]
---

# Decoupled Noise Scheduling

## 定义
在多模态联合扩散模型中，为不同模态（如视频与动作）分别维护独立噪声时间步，允许推理时对各模态设置不同的去噪步数。

## 数学形式

$$
\mathbf{x}_{t_a}^a = (1-t_a)\boldsymbol{\epsilon}_a + t_a\mathbf{a}
$$

$$
\mathbf{x}_{t_v}^v = (1-t_v)\boldsymbol{\epsilon}_v + t_v\mathbf{z}^v
$$

$t_a$ 和 $t_v$ 独立采样，分别控制行动和视频的去噪进度。

## 核心要点
1. 不同模态的信息密度和噪声敏感性不同，应允许独立调度
2. 扩散过程早期（少步数）即可捕获高层结构，视频结构信息无需细粒度细化
3. Efficient-WAM 中视频用 2-5 步、行动用 10 步，延迟从 430ms 降至 139ms

## 代表工作
- [[Efficient-WAM]]: 非对称去噪调度，视频分支用 4 步最优

## 相关概念
- [[Flow Matching]]
- [[Diffusion Model]]
- [[World-Action Model]]
