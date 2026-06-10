---
type: concept
aliases: [变分自编码器, VAE, Variational Autoencoder]
---

# Variational Autoencoder

## 定义

一种生成模型，通过变分推断将输入编码到连续的概率潜空间（而非确定性向量），再从潜空间采样重建输入，同时用 KL 散度约束潜空间分布。

## 数学形式

**编码**：$q_\phi(z | x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))$

**重参数化采样**：$z = \mu + \sigma \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$

**训练目标（ELBO）**：

$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))
$$

## 核心要点

1. **重参数化技巧**: 将随机采样转化为确定性加噪声，使梯度可以流过采样操作
2. **潜空间连续性**: 相近的潜向量解码出相近的输出，便于插值和迁移
3. **KL 散度正则**: 约束潜空间接近标准正态分布，防止过拟合和孔洞问题
4. **条件 VAE**: 加入条件输入 $c$，变为 $q_\phi(z | x, c)$，用于条件生成

## 代表工作

- [[HiMem-WAM]]: 用条件 VAE 将观测和光流编码为低级潜动作 $z^l_t$

## 相关概念

- [[KL Divergence]]
- [[Latent Action]]
- [[Optical Flow]]
