---
type: concept
aliases: [编码器-预测器架构, Encoder-Predictor Architecture]
---

# Encoder-Predictor

## 定义

世界模型的标准两阶段架构：编码器将高维观测（如图像像素）映射到紧凑潜在表示，预测器在潜空间中学习动力学模型。

## 数学形式

$$
z_t = E(o_t), \quad z_{t+1} = P(z_t, a_t)
$$

其中 $E: \mathbb{R}^n \to \mathbb{R}^D$ 为编码器，$P: \mathbb{R}^D \times \mathbb{R}^A \to \mathbb{R}^D$ 为动力学预测器。

## 核心要点

1. 将感知（编码）与动力学学习（预测）解耦，便于独立优化
2. 潜空间压缩降低预测复杂度，使长时序规划更高效
3. 编码器可以是冻结预训练模型（如 DINO-WM）或端到端训练（如 LeWM）
4. 预测器通常为 Transformer 或 MLP，学习 $z_t + a_t \to z_{t+1}$ 的映射

## 代表工作

- [[DINO-WM]]: 冻结 DINOv2 编码器 + ViT 预测器
- [[PLDM]]: JEPA 风格编码器-预测器，多正则项稳定训练
- [[LeWM]]: 端到端从像素训练的极简编码器-预测器
- [[StableWorldModel]]: 对多种编码器-预测器架构的统一评估平台

## 相关概念

- [[JEPA]]
- [[World Model]]
- [[Model Predictive Control]]
