---
type: concept
aliases: [Epps-Pulley Normality Test]
---

# Epps-Pulley Test

## 定义
一种基于经验特征函数与标准正态特征函数差距的单变量正态性检验，给出可微的统计量，适合作为深度学习正则项。

## 数学形式
$$
T(h) = \int \big| \hat\varphi_h(t) - e^{-t^2/2}\big|^2 \, w(t)\, dt
$$
其中 $\hat\varphi_h$ 是样本 $\{h_i\}$ 的经验特征函数，$w$ 是高斯权重。

## 核心要点
1. 闭式可微，便于反传。
2. 比 KS、Shapiro-Wilk 更适合 mini-batch SGD 场景。
3. 在 [[SIGReg]] 中对每个一维投影评估。

## 代表工作
- [[LeWM]]：用作 SIGReg 的内核检验

## 相关概念
- [[SIGReg]]
- [[Cramér-Wold 定理]]
