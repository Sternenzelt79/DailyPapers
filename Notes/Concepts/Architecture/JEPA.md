---
type: concept
aliases: [Joint-Embedding Predictive Architecture, 联合嵌入预测架构]
---

# JEPA (Joint-Embedding Predictive Architecture)

## 定义
LeCun 提出的自监督表征/世界模型范式：在 latent 空间里做"以条件 $c$ 预测 target latent"的预测，不重建像素。

## 数学形式
$$
\mathcal{L}_{\text{JEPA}} = \big\| \text{pred}_\phi(\text{enc}_\theta(x), c) - \text{enc}_\theta(y) \big\|_2^2 + \text{regularizer}
$$

## 核心要点
1. 表示学习与生成解耦：不需要在像素空间重建。
2. 容易塌缩，需要正则项（VICReg、SIGReg）或非对称机制（EMA target、stop-grad）。
3. 是 LeCun "world model" 蓝图的核心组件。

## 代表工作
- [[I-JEPA]]：图像 JEPA
- [[V-JEPA]]：视频 JEPA
- [[LeWM]]：端到端从像素的 JEPA 世界模型，仅 2 项 loss
- [[DINO-WM]]：在冻结 DINO 特征上做 JEPA 预测

## 相关概念
- [[Representation Collapse]]
- [[VICReg]]
- [[SIGReg]]
- [[World Model]]
