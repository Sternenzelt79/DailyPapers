---
type: concept
aliases: [时间抽象, 时序抽象, Temporal Abstraction]
---

# Temporal Abstraction

## 定义

在强化学习和机器人策略中，将连续动作序列抽象为更高层级的时间单元（选项、技能、子目标），使高层策略以低频率做规划决策，低层策略以高频率执行具体动作。

## 数学形式

**选项框架（Options Framework）**：
$$
o = (\mathcal{I}, \pi, \beta), \quad \text{其中 } \mathcal{I} \subseteq \mathcal{S},\; \pi: \mathcal{S} \times \mathcal{A} \to [0,1],\; \beta: \mathcal{S} \to [0,1]
$$

$\mathcal{I}$ 为起始集合，$\pi$ 为选项内策略，$\beta$ 为终止函数。

## 核心要点

1. **层级分解**: 将长时域任务分解为更易学习的短时域技能序列
2. **信用分配**: 缓解长时域任务中的信用分配难题
3. **探索效率**: 通过技能复用提高样本效率
4. **边界检测**: 如何自动发现技能转换边界是核心挑战（见 [[Boundary Scoring]]）

## 代表工作

- [[HiMem-WAM]]: 通过动态分块发现技能边界，实现分层潜动作表示

## 相关概念

- [[Hierarchical Policy]]
- [[Skill Latent]]
- [[Dynamic Chunking]]
- [[Boundary Scoring]]
