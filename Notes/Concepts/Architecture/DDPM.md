---
type: concept
aliases: [Denoising Diffusion Probabilistic Models, 去噪扩散概率模型]
---

# DDPM

## 定义
Denoising Diffusion Probabilistic Models：通过逐步向数据加噪（前向过程）再训练神经网络逐步去噪（反向过程）来学习数据分布的生成模型。

## 数学形式
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

## 核心要点
1. 前向过程：T 步马尔可夫链，逐步向样本加高斯噪声，直到纯噪声
2. 反向过程：训练 UNet 预测每步的噪声 $\epsilon_\theta$，通过迭代去噪生成样本
3. 目标函数：简化为 $L_{simple} = E[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$
4. 采样慢（需要 1000 步），后续工作（DDIM、DPM-Solver）大幅加速

## 代表工作
- [[DDPM]]: Ho et al. 2020，奠定扩散生成模型现代范式
- [[LDM]]: 在隐空间运行 diffusion，大幅降低计算成本

## 相关概念
- [[Diffusion Model]]
- [[LDM]]
- [[DiT]]
- [[Flow Matching]]
