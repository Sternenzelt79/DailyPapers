---
type: concept
aliases: [Mixture of Layers]
---

# MoLe

## 定义
Mixture of Layers，一种动态层路由方法，根据输入内容自适应地跳过或激活特定层，实现计算-精度权衡。

## 核心要点
1. 类似 MoE 对专家的路由机制，但作用于网络深度维度
2. 允许模型对简单样本使用浅层，对复杂样本使用更多层
3. 需要额外的路由器训练，与 [[CKA]]-based 静态剪枝不同

## 代表工作
- [[CLP]]: 对比基线，显示静态剪枝优于动态路由

## 相关概念
- [[CKA]]
- [[SpecPrune]]
