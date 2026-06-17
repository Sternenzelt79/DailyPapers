---
type: concept
aliases: [VQ, Vector Quantization, 向量量化]
---

# VQ（Vector Quantization / 向量量化）

## 定义
向量量化将连续向量映射到离散 codebook 中最近邻的码字，是 VQ-VAE、DALL-E、VQGAN 等离散表示学习模型的核心操作。

## 数学形式
给定连续 encoder 输出 $z_e$，码字查找：
$$z_q = e_k, \quad k = \arg\min_j \| z_e - e_j \|_2$$

损失函数（包含 commitment loss）：
$$\mathcal{L} = \| x - \text{dec}(z_q) \|^2 + \| \text{sg}(z_e) - e \|^2 + \beta \| z_e - \text{sg}(e) \|^2$$

直通梯度（straight-through estimator）：
$$\frac{\partial \mathcal{L}}{\partial z_e} \approx \frac{\partial \mathcal{L}}{\partial z_q}$$

## 核心要点
1. 离散瓶颈：把连续 latent 压缩成 K 个码字的索引序列，可用自回归建模
2. Straight-through estimator 绕过不可微的 argmin 操作
3. Codebook collapse（部分码字永不更新）是常见问题，需 EMA 更新或 codebook reset
4. RVQ（残差 VQ）分层量化提高码字利用率，用于 Encodec、SoundStream

## 代表工作
- VQ-VAE (van den Oord et al., 2017)
- VQGAN (Esser et al., 2021)
- RVQ / Encodec (Défossez et al., 2022)

## 相关概念
- [[VQ-VAE]]
- [[VQGAN]]
- [[Codebook]]
