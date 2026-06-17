---
type: concept
aliases: [LAM, 潜在动作模型]
---

# Latent Action Model

## 定义

在视觉特征的潜在空间中学习动作表示的模型框架：逆动力学编码器从观测转移中推断潜在动作，解码器用潜在动作预测未来状态，无需显式的像素级重建。

## 数学形式

$$
z \sim q_\phi(z|u, u_T), \quad \tilde{u}_T = \text{Decoder}_\omega(u, z)
$$

训练目标（含 KL 正则化）：

$$
\mathcal{L}_{\text{LAM}} = \|\tilde{u}_T - u_T\|_2^2 + \beta \cdot D_{\text{KL}}(q_\phi(z|u,u_T) \| \mathcal{N}(0,I))
$$

## 核心要点

1. **具身无关**：潜在动作 $z$ 编码视觉状态转移的"意图"，可跨不同具身迁移
2. **无需动作标签**：仅从观测对 $(o_t, o_{t+H})$ 学习，适合无动作标签的视频数据
3. **与 VLA 集成**：蒸馏使 VLA 策略先验学会预测 $\hat{z}$，从而驱动世界模型

## 相关概念

- [[Latent Action]]
- [[逆动力学编码器]]
- [[World-Action Model]]

## 代表工作

- [[LaWAM]]: Stage 1 预训练 LAM，学习 LaWM；Stage 2 将 LaWM 集成进 VLA
- [[LAPA]]: 早期潜在动作模型
- [[UniVLA]]: 基于潜在动作的跨具身 VLA
