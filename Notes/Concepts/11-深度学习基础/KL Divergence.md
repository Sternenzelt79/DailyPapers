---
type: concept
aliases: [KL 散度, 相对熵, Kullback-Leibler Divergence, KL Divergence]
---

# KL Divergence

## 定义

衡量两个概率分布 $P$ 与 $Q$ 之间差异的非对称度量，等于用分布 $Q$ 近似 $P$ 时的额外信息量（比特数）。

## 数学形式

$$
D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]
$$

对于两个高斯分布 $P = \mathcal{N}(\mu_1, \sigma_1^2)$ 和 $Q = \mathcal{N}(0, I)$：

$$
D_{\text{KL}}(P \| Q) = \frac{1}{2}\sum_j \left(\mu_{1,j}^2 + \sigma_{1,j}^2 - \ln \sigma_{1,j}^2 - 1\right)
$$

## 核心要点

1. **非对称性**: $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$（非度量）
2. **非负性**: $D_{\text{KL}}(P \| Q) \geq 0$，当且仅当 $P = Q$ 时等号成立
3. **VAE 正则化**: 在变分自编码器中，KL 散度约束后验 $q(z|x)$ 接近先验 $p(z) = \mathcal{N}(0, I)$
4. **信息论解释**: 等价于 $P$ 相对于 $Q$ 的超额编码长度

## 代表工作

- [[HiMem-WAM]]: 低级潜动作分词器训练损失中的 KL 正则项 $\beta D_{\text{KL}}(q_\phi(z^l_t|c_t) \| \mathcal{N}(0, I))$

## 相关概念

- [[Variational Autoencoder]]
- [[Latent Action]]
