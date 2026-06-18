---
type: concept
aliases: [Energy-Based JEPA, 能量型联合嵌入预测架构]
---

# EB-JEPA (Energy-Based JEPA)

## 定义
EB-JEPA 是基于能量函数的联合嵌入预测架构变体，使用 VICReg 正则化训练潜在表示，通过能量最小化评估预测状态的一致性。

## 数学形式
使用 VICReg 损失约束：
$$\mathcal{L}_{\text{VICReg}} = \lambda_v \mathcal{L}_{\text{var}} + \lambda_c \mathcal{L}_{\text{cov}} + \lambda_i \mathcal{L}_{\text{inv}}$$

## 核心要点
1. 基于 JEPA 框架，加入能量函数建模预测不确定性
2. VICReg 防止表示坍缩（variance + covariance + invariance）
3. 消融实验显示：长 horizon 下累积预测误差导致相位对齐匹配不可靠

## 代表工作
- [[DREAM-Chunk]]: 消融实验中评估了 EB-JEPA 作为辅助世界模型，发现长 horizon 性能退化

## 相关概念
- [[JEPA]]
- [[潜在世界模型]]
- [[LeWorldModel]]
- [[R2-Dreamer]]
