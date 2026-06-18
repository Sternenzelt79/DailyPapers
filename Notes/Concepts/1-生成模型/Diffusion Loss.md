---
type: concept
aliases: [扩散损失, Flow Matching Loss, 速度场预测损失]
---

# Diffusion Loss

## 定义

Diffusion Loss 是扩散模型或 Flow Matching 模型的核心训练目标，要求神经网络预测噪声与数据之间的速度场（velocity field），使生成过程能够沿正确方向演化。

## 数学形式

**Flow Matching 形式**（PAIWorld 使用）：

$$
\mathcal{L}_{\text{diff}} = \mathbb{E}_{s,\varepsilon}\!\left[\left\|\, u_\theta(z_s, s) - (\varepsilon - z_0)\,\right\|_2^2\right]
$$

其中 $z_s = (1-s)z_0 + s\varepsilon$ 为线性插值路径上的带噪样本。

**DDPM 形式**（噪声预测）：

$$
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t,\varepsilon}\!\left[\left\|\varepsilon - \varepsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon, t)\right\|_2^2\right]
$$

## 核心要点

1. **速度场 vs 噪声预测**: Flow Matching 预测速度场 $v = \varepsilon - z_0$；DDPM 预测噪声 $\varepsilon$
2. **简单高效**: $L_2$ 损失直接优化，训练稳定
3. **条件生成**: 通过 classifier-free guidance 等方式引入条件信号

## 代表工作

- [[Flow Matching]]: Flow Matching 训练范式
- [[PAIWorld]]: 使用 Flow Matching 扩散损失作为基础生成目标，配合 REPA 损失联合优化

## 相关概念

- [[Flow Matching]]
- [[Diffusion Transformer]]
- [[VAE]]
