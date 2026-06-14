---
type: concept
aliases: [边界得分, 边界检测, Boundary Scoring, Boundary Detection]
---

# Boundary Scoring

## 定义

在序列分割任务中，通过计算相邻 token 之间的语义距离（如余弦不相似度）来给每个位置赋予一个"是否为边界"的得分，用于自动发现序列的分段点。

## 数学形式

$$
r_i = \frac{1}{2}\left(1 - \hat{q}_{i-1}^\top \hat{k}_i\right) \in [0, 1]
$$

$$
b_i = \mathbf{1}[r_i \geq \delta]
$$

其中 $\hat{q}, \hat{k}$ 为归一化后的查询/键向量，$\delta$ 为边界判断阈值，可固定或可学习。

## 核心要点

1. **无监督边界发现**: 仅依赖序列内容的相似度，不需要人工标注边界
2. **余弦距离**: 对向量幅值不敏感，只关注方向变化，更鲁棒
3. **阈值控制**: $\delta$ 控制边界粒度，较小的 $\delta$ 产生更细粒度的分段
4. **监督可用时**: 发现的边界可以映射回原始时间步作为监督信号训练边界预测器

## 代表工作

- [[HiMem-WAM]]: 用余弦不相似度检测低级潜动作中的技能转换边界，用于动态分块和记忆写门触发

## 相关概念

- [[Dynamic Chunking]]
- [[Memory-Gated Module]]
- [[Temporal Abstraction]]
- [[Skill Latent]]
