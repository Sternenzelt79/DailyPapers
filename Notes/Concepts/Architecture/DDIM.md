---
type: concept
aliases: [Denoising Diffusion Implicit Models, DDIM 采样, 隐式扩散去噪]
---

# DDIM（Denoising Diffusion Implicit Models）

## 定义

DDIM 是 DDPM 的加速推理变体，通过将马尔科夫扩散过程推广为**非马尔科夫隐式过程**，在保持生成质量的同时将采样步数从 1000 步压缩到 10-50 步。

## 数学形式

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \varepsilon_\theta(x_t, t) + \sigma_t \varepsilon
$$

其中 $\sigma_t = 0$ 时退化为完全确定性（DDIM）；$\sigma_t = \sqrt{(1-\bar{\alpha}_{t-1})/(1-\bar{\alpha}_t)} \cdot \sqrt{1-\bar{\alpha}_t/\bar{\alpha}_{t-1}}$ 时等价于 DDPM。

## 核心要点

1. **非马尔科夫前向过程**：逆扩散步骤不依赖于全部历史，可跳步；
2. **确定性采样**（$\sigma_t=0$）：相同初始噪声生成相同输出，利于插值和编辑；
3. **CFG（Classifier-Free Guidance）兼容**：scale 参数控制条件强度 vs 多样性；
4. **在机器人策略中的应用**：10 步 DDIM + CFG 是扩散动作专家（如 CogACT、MemoryVLA++）的标准推理配置。

## 代表工作

- [[MemoryVLApp]]: 动作专家采用 10 步 DDIM，CFG scale=1.5
- [[CogACT]]: 扩散动作专家推理使用 DDIM
- [[Diffusion Policy]]: 机器人策略的扩散推理基础

## 相关概念

- [[扩散模型]]
- [[Diffusion Transformer]]
- [[Diffusion Policy]]
