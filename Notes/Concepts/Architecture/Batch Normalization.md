---
type: concept
aliases: [BatchNorm, BN, Batch Normalization]
---

# Batch Normalization (BN)

## 定义
按 batch 维度对特征做零均值单位方差归一化，再用可学 $\gamma,\beta$ 缩放平移。Ioffe & Szegedy, 2015。

## 数学形式
$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}},\quad y_i = \gamma \hat{x}_i + \beta
$$

## 核心要点
1. 训练加速、起到隐式正则。
2. 推理时使用滑动均值/方差。
3. 在 SSL / JEPA 头里用来打散 latent 分布，避免崩塌。

## 代表工作
- [[LeWM]]：projection MLP 中用 BN
- ResNet 系列

## 相关概念
- [[Layer Normalization]]
- [[Representation Collapse]]
