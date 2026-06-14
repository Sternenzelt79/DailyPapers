---
type: concept
aliases: [时间运动蒸馏, 帧间差分蒸馏]
---

# Temporal Motion Distillation

## 定义
知识蒸馏的一种变体，通过对齐教师与学生模型的帧间隐藏状态差分来显式传递时序运动建模能力。

## 数学形式

$$
\mathcal{L}_{\text{mot}} = \frac{1}{|\mathcal{A}_{\text{mot}}|}\sum_{l \in \mathcal{A}_{\text{mot}}}\mathbb{E}_f\left[1 - \cos\left(\Delta\overline{h}_{s,f}^l, \Delta\overline{h}_{t,f}^{\tau(l)}\right)\right]
$$

其中 $\Delta\overline{h}^l_f$ 为第 $l$ 层相邻帧 $f$ 的隐藏状态差分（空间平均后）。

## 核心要点
1. 比静态隐藏状态蒸馏更有效地传递运动动态信息
2. 帧间差分天然编码了运动方向、速度等时序信息
3. 与隐藏状态蒸馏（$\mathcal{L}_{\text{hid}}$）联合使用，两者互补

## 代表工作
- [[Efficient-WAM]]: 在紧凑视频专家蒸馏中引入时间运动蒸馏

## 相关概念
- [[Knowledge Distillation]]
- [[Video Diffusion Model]]
