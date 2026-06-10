---
type: concept
aliases: [分层潜动作, 层级潜动作, Hierarchical Latent Action]
---

# Hierarchical Latent Action

## 定义

将机器人动作空间组织成多个层级的潜变量表示：低级潜动作编码短时域运动原语（逐步执行细节），高级技能潜变量编码长时域语义技能（任务分解单元），两者联合实现短时执行与长时规划的桥梁。

## 数学形式

$$
\pi(a_{t:t+K} | s, \ell, M) = \int p(a | Z^l, s) \cdot p(Z^l | z^h, s, M) \cdot p(z^h | s, \ell, M)\, dZ^l\, dz^h
$$

低级：$z^l_t \sim q_\phi(z^l_t | \text{obs}, \text{flow})$（变分采样）

高级：$z^h_j = \text{Pool}(\{z^l_i\}_{i \in \mathcal{I}_j})$（技能边界内的池化）

## 核心要点

1. **双层分离**: 低级处理"怎么动"，高级处理"做什么"
2. **光流监督**: 低级潜动作通过光流重建进行无监督预训练
3. **边界对齐**: 高低级之间的边界发现是关键（见 [[Boundary Scoring]]）
4. **记忆扩展**: 高级技能潜变量可直接与外部记忆接口，低级潜动作不需要记忆

## 代表工作

- [[HiMem-WAM]]: 分层潜动作的完整实现，结合记忆门控模块

## 相关概念

- [[Latent Action]]
- [[Skill Latent]]
- [[Hierarchical Policy]]
- [[Dynamic Chunking]]
- [[Memory-Gated Module]]
