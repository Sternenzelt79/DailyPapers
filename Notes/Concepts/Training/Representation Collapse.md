---
type: concept
aliases: [表示塌缩, Feature Collapse]
---

# Representation Collapse

## 定义
自监督学习中 encoder 退化到平凡解（常数或低秩 latent），导致表示无信息。

## 核心要点
1. **完全塌缩**：所有输入映射到同一向量。
2. **维度塌缩**：latent 实际只占据低维子空间。
3. 缓解手段：对比学习、stop-gradient（[[BYOL]]/[[SimSiam]]）、EMA target、协方差正则（[[VICReg]]）、分布正则（[[SIGReg]]）。

## 代表工作
- [[VICReg]]：variance + covariance 正则
- [[SIGReg]] / [[LeWM]]：单一各向同性高斯正则
- [[BYOL]]：非对称 + EMA

## 相关概念
- [[JEPA]]
- [[VICReg]]
- [[SIGReg]]
