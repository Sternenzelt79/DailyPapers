---
type: concept
aliases: [潜在动作 Tokenizer, Latent Action Tokenization]
---

# Latent Action Tokenizer

## 定义

从无标注观测序列中无监督学习紧凑动作表征的模块，通常由逆向动态模型（IDM）和前向动态模型（FDM）耦合组成，将相邻视觉帧的变化压缩为潜在动作 token。

## 数学形式

$$
\ell_t = q_\phi(z_t, z_{t+1}), \quad \hat{z}_{t+1} = f_\psi(z_t, \ell_t)
$$

训练目标：

$$
\mathcal{L}_{fwd} = \sum_t \|\hat{z}_{t+1} - z_{t+1}\|_2^2, \quad \mathcal{L}_{cons} = \sum_t \|\hat{z}_t - z_t\|_2^2
$$

## 核心要点

1. **无监督**：不需要真实动作标签，从视频序列自监督学习
2. IDM 提取动作，FDM 验证动作的预测能力（防止内容泄漏）
3. 视觉空间的选择直接影响潜在动作的质量（语义空间 vs 像素空间）
4. 两阶段训练：先在视频上预训练，再适配到有真实动作标签的机器人数据

## 代表工作

- [[RepWAM]]: 在语义对齐视觉空间内学习潜在动作，IDM 损失更低，迁移性更强
- [[Latent Action]]: 潜在动作的上层概念

## 相关概念

- [[Latent Action]]
- [[Inverse Dynamics Model]]
- [[Forward Dynamics Model]]
- [[RepViTok]]
- [[World-Action Model]]
