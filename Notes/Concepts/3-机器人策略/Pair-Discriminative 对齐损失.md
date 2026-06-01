---
type: concept
aliases: [PD Loss, Pair-Discriminative Loss, 对对识别损失]
---

# Pair-Discriminative 对齐损失（PD Loss）

## 定义

Pair-Discriminative（PD）损失是一种对比 hinge 损失，要求**配对的跨具身特征之间的距离**小于该样本与批次内其他配对样本的平均距离，增强配对样本的区分性。

## 数学形式

$$
\mathcal{L}_{PD} = \mathbb{E}_{(H,R)\sim\mathcal{D}_p} \sum_{\alpha\in\{R\to H,\, H\to R\}} \lambda_\alpha \left[m_t + d(f^R, f^H) - \bar{d}^\alpha\right]_+
$$

其中 $\bar{d}^\alpha$ 为当前样本与批次内其他配对的平均余弦距离，$m_t$ 为 margin 超参数。

## 核心要点

1. **双向对比**：同时考虑 R→H 和 H→R 两个对比方向，全面约束配对关系
2. **批次级别的负样本**：利用批次内其他配对作为隐式负样本，不需要显式构造负对
3. **配对识别**：本质是要求模型能够"识别出"哪些人机帧是真正配对的
4. **与 SR 互补**：SR 负责缩小绝对距离，PD 负责增强相对区分度

## 代表工作

- [[HARP-VLA]]: 提出 SRPD = SR + PD 的组合损失，在跨具身检索 R@1 上比单独 SR 再提升约 4 个点

## 相关概念

- [[Source-Relative 对齐损失]]
- [[余弦距离]]
- [[潜在动作模型]]
- [[跨具身学习]]
