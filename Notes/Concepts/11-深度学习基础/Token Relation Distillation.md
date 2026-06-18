---
type: concept
aliases: [Token关系蒸馏, 关系蒸馏, token-relation distillation]
---

# Token Relation Distillation

## 定义

Token Relation Distillation 是一种知识蒸馏策略，通过对齐学生网络与教师网络的 token 间相似度矩阵（而非直接对齐特征值），将教师的结构化知识迁移到学生模型，绕过特征空间维度不匹配的问题。

## 数学形式

**相似度矩阵计算**：

$$
S(F)_{i,a} = \frac{f_i^\top f_a}{\|f_i\| \cdot \|f_a\|}, \quad a \in \mathcal{A}
$$

**蒸馏损失**：

$$
\mathcal{L}_{\text{distill}} = \text{SmoothL1}(S^{\text{student}}, S^{\text{teacher}})
$$

## 核心要点

1. **空间不变性**: 只对齐相对关系（余弦相似度），不要求教师和学生特征维度相同
2. **结构化知识**: 保留了教师特征空间中的拓扑结构（哪些 token 相似/不相似）
3. **与特征蒸馏对比**: 特征蒸馏需要特征对齐层（projection head），关系蒸馏更轻量且更灵活
4. **Anchor 采样**: 常配合 anchor 采样将 $O(N^2)$ 复杂度降为 $O(M \cdot K)$

## 代表工作

- [[Latent 3D-REPA]]: 在 PAIWorld 中使用 token 关系蒸馏提供 3D 几何监督
- [[REPA]]: Token 关系蒸馏的基础框架

## 相关概念

- [[REPA]]
- [[Latent 3D-REPA]]
- [[Depth Anything 3]]
