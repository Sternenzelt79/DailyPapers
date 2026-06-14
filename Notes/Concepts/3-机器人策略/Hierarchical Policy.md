---
type: concept
aliases: [分层策略, 层级策略, Hierarchical Policy]
---

# Hierarchical Policy

## 定义

将复杂任务分解为多个层级的子目标或子技能，高层负责规划（选择技能/子目标），低层负责执行（生成具体动作），两层之间通过接口（如潜变量）通信。

## 数学形式

$$
\pi(a | s, g) = \int \pi_{\text{low}}(a | s, z) \cdot \pi_{\text{high}}(z | s, g)\, dz
$$

其中 $z$ 是高层传递给低层的子目标或技能潜变量，$g$ 是全局任务目标。

## 核心要点

1. **时间抽象**: 高层以较低频率做决策（选技能），低层以较高频率执行动作
2. **接口设计**: 高低层之间的接口（子目标、潜变量、选项）是关键设计点
3. **监督信号**: 可对高低层分别设计监督信号，降低端到端学习难度
4. **长时域效果**: 在长时域任务中显著优于扁平策略，因高层可跨越局部最优

## 代表工作

- [[HiMem-WAM]]: 将策略分解为技能规划器 + 运动执行器 + 动作解码器，高层使用 Qwen3-VL

## 相关概念

- [[Temporal Abstraction]]
- [[Skill Latent]]
- [[Hierarchical Latent Action]]
- [[World-Action Model]]
