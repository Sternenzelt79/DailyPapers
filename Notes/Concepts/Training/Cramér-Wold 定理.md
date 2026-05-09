---
type: concept
aliases: [Cramer-Wold Theorem, Cramér-Wold]
---

# Cramér-Wold 定理

## 定义
一个 $\mathbb{R}^d$ 上的概率分布由其所有一维线性投影的分布唯一确定。

## 数学形式
$$
P_X = P_Y \iff \forall u\in\mathbb{R}^d,\; P_{\langle u,X\rangle} = P_{\langle u,Y\rangle}
$$

## 核心要点
1. 把高维分布匹配问题降到 1D。
2. 实践中只需对若干随机方向做匹配即可近似。
3. 是 [[SIGReg]] 之类 sketched 正则项的理论依据。

## 代表工作
- [[LeWM]]：用 Cramér-Wold + 1D 正态性检验构造 SIGReg。

## 相关概念
- [[SIGReg]]
- [[Epps-Pulley Test]]
