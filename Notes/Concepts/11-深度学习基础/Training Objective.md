---
type: concept
aliases: [训练目标, 联合训练目标, joint training objective]
---

# Training Objective

## 定义

训练目标（Training Objective）是深度学习模型优化的损失函数，通常由主任务损失与辅助正则化/蒸馏损失的加权组合构成，指导模型同时学习多个目标。

## 数学形式

通用形式：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + \sum_i \lambda_i \mathcal{L}_{\text{aux},i}
$$

PAIWorld 中的具体实例：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda \cdot \mathcal{L}_{\text{REPA}}, \quad \lambda = 0.5
$$

## 核心要点

1. **多目标权衡**: 权重系数 $\lambda$ 控制主任务与辅助任务之间的平衡
2. **常见辅助损失**: 蒸馏损失、对比损失、正则化损失、一致性损失
3. **权重调优**: $\lambda$ 通常通过消融实验确定，对最终性能影响显著

## 代表工作

- [[PAIWorld]]: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + 0.5 \cdot \mathcal{L}_{\text{REPA}}$，平衡生成质量与 3D 几何对齐

## 相关概念

- [[Diffusion Loss]]
- [[Latent 3D-REPA]]
- [[Flow Matching]]
