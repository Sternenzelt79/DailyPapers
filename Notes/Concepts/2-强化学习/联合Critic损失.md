---
type: concept
aliases: [Joint Critic Loss, 联合Critic损失, 双头Critic损失]
---

# 联合Critic损失

## 定义

联合 Critic 损失是 HABC 中同时训练存活性头和效率头的组合损失函数，两个分支在不同数据子集上分别计算损失后求和。

## 数学形式

$$
\mathcal{L}_{\text{critic}} = \mathbb{E}_{\mathcal{D}^{\text{lab}}_{\text{auto}}}[\text{BCE}(z_v, y)] + \mathbb{E}_{\mathcal{D}_{\text{succ}}}[\text{Huber}(\hat{V}_e, y_e)]
$$

## 核心要点

1. **数据分区**: 存活性头使用所有有标注数据（$\mathcal{D}^{\text{lab}}_{\text{auto}}$），效率头仅使用成功轨迹（$\mathcal{D}_{\text{succ}}$）
2. **异构损失**: BCE 用于概率分类，Huber 用于步数回归，两种损失量纲不同但直接求和
3. **梯度互不干扰**: 两头独立线性层，共享 $\phi$ 但对同一参数的梯度来源不同数据分布
4. **端到端**: Critic 和 Actor 采用同一主干 $\phi$，联合损失使 Critic 训练也反向传播到特征层

## 代表工作

- [[HABC]]: 联合 Critic 损失驱动双头价值网络的联合优化

## 相关概念

- [[双头Critic]]
- [[二元交叉熵]]
- [[Huber 损失]]
- [[存活性（Viability）]]
- [[效率（Efficiency）]]
