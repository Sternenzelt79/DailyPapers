---
type: concept
aliases: [Variance-Invariance-Covariance Regularization]
---

# VICReg

## 定义
自监督表征学习的正则化方法，由 variance（保证每维方差不塌）、invariance（增广不变性）、covariance（去相关）三项组成。

## 数学形式
$$
\mathcal{L}_{\text{VICReg}} = \lambda \, s(Z, Z') + \mu \, [v(Z) + v(Z')] + \nu \, [c(Z) + c(Z')]
$$

## 核心要点
1. 不需要 EMA / stop-grad 的对称结构。
2. 三项各有权重，超参较多。
3. 被 [[SIGReg]] 在 [[LeWM]] 中用单一项替代。

## 代表工作
- VICReg 原作 (Bardes et al. 2022)
- [[PLDM]]：在世界模型中堆叠 VICReg 等正则
- [[LeWM]]：用 SIGReg 取而代之

## 相关概念
- [[SIGReg]]
- [[Representation Collapse]]
- [[JEPA]]
