---
type: concept
aliases: [潜动作, 潜在动作, Latent Action]
---

# Latent Action

## 定义

将机器人动作或运动意图编码到连续的低维潜空间中得到的表示，可以从视频/光流等无监督信号中学习，用于桥接视觉观测与动作执行。

## 数学形式

$$
z^l_t = \text{Enc}(o_t, o_{t+1}, \Phi_t), \quad a_t = \text{Dec}(z^l_t, o_t)
$$

其中 $\Phi_t$ 为光流（可选），$z^l_t$ 为低级潜动作，Enc/Dec 通常为变分自编码器结构。

## 核心要点

1. **无监督学习**: 可从视频光流等信号中学习，不强依赖动作标注
2. **运动先验**: 学到的潜动作编码了任务完成的运动模式，比原始像素更鲁棒
3. **可迁移性**: 潜动作作为预训练特征，可在不同场景/扰动下迁移
4. **与技能的关系**: 低级潜动作可进一步聚合为高级技能潜变量（[[Skill Latent]]）

## 代表工作

- [[HiMem-WAM]]: 通过变分分词器将光流编码为低级潜动作，再层级化为技能潜变量
- [[潜在动作模型]]: 从视频中无监督学习潜动作空间

## 相关概念

- [[Hierarchical Latent Action]]
- [[Optical Flow]]
- [[Variational Autoencoder]]
- [[World-Action Model]]
