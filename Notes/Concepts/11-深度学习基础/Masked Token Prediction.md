---
type: concept
aliases: [Masked Token Prediction, MTP, Masked Prediction, masked token modeling]
---

# Masked Token Prediction

## 定义
一种自回归或非自回归生成策略：随机遮蔽序列中的部分 token，训练模型根据可见 token 预测被遮蔽的 token。迭代应用可逐步精化生成质量，常用于离散 token 空间中的序列生成。

## 数学形式

$$
\mathcal{L} = -\sum_{i \in \mathcal{M}} \log P(z_i \mid z_{\backslash \mathcal{M}})
$$

其中 $\mathcal{M}$ 为被遮蔽位置集合，$z_{\backslash \mathcal{M}}$ 为可见 token。

## 核心要点

1. **迭代精化**: 从全遮蔽开始，每轮预测最高置信度的 token，逐步填充完整序列
2. **并行生成**: 比自回归逐 token 生成更快，适合实时应用
3. **与 BERT 的关系**: BERT 用于理解任务；Masked Token Prediction 生成模型（如 MaskGIT）用于生成任务
4. **在运动生成中**: [[SONIC]] 的运动规划器在 [[FSQ]] token 空间中用此方法生成运动序列，推理 ~12ms

## 代表工作
- MaskGIT (Chang et al., 2022): 图像生成
- [[SONIC]]: 在 FSQ latent 空间中用于运动序列生成

## 相关概念
- [[FSQ]]
- [[Autoregressive Policy]]
- [[Transformer]]
