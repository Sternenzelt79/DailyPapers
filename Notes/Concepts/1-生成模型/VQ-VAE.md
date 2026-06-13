---
type: concept
aliases: [Vector Quantized VAE, 向量量化变分自编码器, VQ-VAE, VQVAE]
---

# VQ-VAE（Vector Quantized Variational Autoencoder）

## 定义

用离散码本（codebook）替代连续潜变量的变分自编码器，通过向量量化将连续编码器输出映射到有限的离散嵌入向量集合，实现离散潜在表示学习。

## 数学形式

编码器输出 $\mathbf{z}_e$，量化到码本 $\{e_k\}_{k=1}^K$ 中最近的向量：

$$
\mathbf{z}_q = e_k, \quad k = \arg\min_j \|\mathbf{z}_e - e_j\|_2
$$

训练损失（含 stop-gradient）：

$$
\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 + \|\text{sg}[\mathbf{z}_e] - \mathbf{z}_q\|_2^2 + \beta\|\mathbf{z}_e - \text{sg}[\mathbf{z}_q]\|_2^2
$$

## 核心要点

1. **离散潜变量**: 与标准 VAE 的连续高斯潜变量不同，VQ-VAE 使用离散码本索引，适合建模语言、动作等离散结构。
2. **直通梯度（Straight-Through Estimator）**: 量化操作不可微，训练时将解码器梯度直接复制给编码器，绕过离散不可微问题。
3. **Commitment Loss**: 第三项 $\beta\|\mathbf{z}_e - \text{sg}[\mathbf{z}_q]\|_2^2$ 防止编码器输出偏离码本太远，稳定训练。
4. **Codebook 更新**: 码本向量通过 EMA（指数移动平均）或梯度下降更新，防止码本坍缩（codebook collapse）。

## 代表工作

- [[LARA]]: 在 LAM（潜在动作模型）中使用 VQ-VAE 对逆向动力学编码的潜在动作进行量化，量化潜变量用于与 VLA DiT 特征对齐
- [[LAPO]]: 使用 VQ-VAE 从无标注视频中离散化潜在动作，作为机器人策略的伪动作标签
- [[LAPA]]: 同样基于 VQ-VAE 框架提取潜在动作

## 相关概念

- [[变分自编码器|VAE]]: VQ-VAE 的前身，使用连续高斯潜变量
- [[潜在动作模型|LAM]]: 使用 VQ-VAE 量化动作表示的机器人学习框架
- [[逆向动力学模型|IDM]]: VQ-VAE 编码器在机器人策略中的对应角色
