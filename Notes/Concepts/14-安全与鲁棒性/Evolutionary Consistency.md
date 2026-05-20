---
type: concept
aliases: [EC, Evolutionary Consistency, 演化一致性]
---

# Evolutionary Consistency（演化一致性）

## 定义
在扩散/流匹配策略的去噪过程中，约束不同时间步的速度场预测保持一致，以抑制动作生成中的表示波动。

## 数学形式

$$
\mathcal{L}_{EC} = \left\|\hat{v}_{clean}^{\tau_1} - \hat{v}_{clean}^{\tau_2}\right\|_2^2
$$

其中 $\tau_1, \tau_2 \sim \mathcal{U}(0,1)$ 为随机采样的两个去噪时间步，$\hat{v}_{clean}^{\tau_i}$ 为对应预测的清洁速度场。

## 核心要点
1. 不增加推理开销（训练阶段约束）
2. 两点监督避免了多点监督的计算冗余
3. 可作为 OC（观测一致性）扰动生成的梯度来源

## 代表工作
- [[RoVLA]]: EC 是三重一致性约束之一，专门针对动作生成过程鲁棒性

## 相关概念
- [[一致性正则化]]
- [[Observational Consistency]]
- [[条件流匹配]]
