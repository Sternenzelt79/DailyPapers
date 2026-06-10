---
type: concept
aliases: [动态分块, 可变长度分割, Dynamic Chunking]
---

# Dynamic Chunking

## 定义

根据序列中相邻元素的语义相似度（如余弦距离），自适应地将序列划分为可变长度的语义片段（chunk），而非使用固定长度的滑动窗口。

## 数学形式

**异质性得分**：
$$
r_i = \frac{1}{2}\left(1 - \hat{q}_{i-1}^\top \hat{k}_i\right)
$$

**边界判断**：
$$
b_i = \mathbf{1}[r_i \geq \delta]
$$

其中 $\hat{q}, \hat{k}$ 为归一化的查询/键向量，$\delta$ 为可学习或固定的阈值。

## 核心要点

1. **自适应粒度**: 语义相似的片段合并为一个 chunk，语义跳变处形成新 chunk
2. **无需固定窗口**: 克服固定分块对任务结构的假设
3. **层级化**: 可在多个层级递归应用，每层产生更高抽象的 token
4. **应用场景**: 语言模型 token 压缩、机器人技能发现、视频分段

## 代表工作

- [[HiMem-WAM]]: 对低级潜动作序列进行动态分块，发现技能边界并生成技能潜变量

## 相关概念

- [[Boundary Scoring]]
- [[Temporal Abstraction]]
- [[Attention-Based Pooling]]
- [[Skill Latent]]
