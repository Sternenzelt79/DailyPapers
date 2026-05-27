---
type: concept
aliases: [Combine-then-Filter策略, 合并再过滤]
---

# Combine-then-Filter

## 定义

一种 Token 选择策略：先将多个候选集合取并集（Combine），再通过多样性优化去除冗余（Filter），最终得到紧凑的任务相关子集。

## 核心要点

1. **两步解耦**: 候选集构建（最大化相关性覆盖）与去冗余过滤（最小化冗余）分开处理，避免分数融合的超参敏感性
2. **并集保全**: 通过并集操作确保不同视角（语义/动作）的关键 Token 均被保留
3. **多样性过滤**: 通常采用 [[最大最小多样性问题|MMDP]] 求解最优多样性子集

## 数学形式

$$
\mathcal{C}_{dual} = \mathcal{C}_{vl} \cup \mathcal{C}_{act}
$$

$$
\tilde{\mathcal{C}} = \arg\max_{\mathcal{C} \subset \mathcal{C}_{dual},\, |\mathcal{C}| = \tilde{M}} \min_{i \neq j \in \mathcal{C}} d(v_i, v_j)
$$

## 代表工作

- [[VLA-Pruner]]: 提出 Combine-then-Filter 策略，用于 VLA Token 剪枝中的语义-动作双候选集合并与去冗余

## 相关概念

- [[最大最小多样性问题]]
- [[视觉 Token 剪枝]]
- [[余弦相似度]]
