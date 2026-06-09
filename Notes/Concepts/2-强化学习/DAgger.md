---
type: concept
aliases: [Dataset Aggregation, 数据集聚合]
---

# DAgger

## 定义

DAgger（Dataset Aggregation）是一种交互式模仿学习算法，通过在策略执行时收集专家纠正数据并聚合到训练集，逐步减少分布偏移（distribution shift）导致的累积误差。

## 数学形式

$$
\mathcal{D}_{i+1} = \mathcal{D}_i \cup \{(s_t, a_t^*) \mid s_t \sim \pi_i\}
$$

其中 $a_t^*$ 为专家在策略 $\pi_i$ 诱导的状态分布下提供的修正动作。

## 核心要点

1. **解决分布偏移**: 标准行为克隆只在专家分布下训练，而 DAgger 在学习策略的状态分布下采集专家标注
2. **交互式数据收集**: 策略上线运行，人类/专家实时提供干预或纠正
3. **渐进改善**: 随迭代轮次增加，策略分布逐渐向专家分布靠近

## 代表工作

- [[FAWAM]]: 用 DAgger 风格收集 20 次人工干预演示训练残差修正器

## 相关概念

- [[Imitation Learning]]
- [[Residual Policy Learning]]
