---
type: concept
aliases: [Representation Visual-Action Tokenizer, 表征视觉-动作 Tokenizer]
---

# RepViTok

## 定义

RepWAM 提出的表征视觉-动作 tokenizer，在共享语义空间内联合学习视觉 token（与冻结基础模型对齐）和潜在动作 token（建模语义状态间的迁移），取代传统重建导向 tokenizer。

## 数学形式

视觉对齐损失：

$$
\mathcal{L}_{align} = \|avg(W_{align}\, z) - avg(G(o))\|_2^2
$$

潜在动作前向动态：

$$
\ell_t = q_\phi(z_t, z_{t+1}), \quad K_t, \delta_t = f_\psi(z_t, \ell_t), \quad \hat{z}_{t+1} = K_t z_t + \delta_t
$$

## 核心要点

1. **双目标视觉编码器**：重建损失（像素保真）+ 特征对齐损失（语义）
2. **耦合 IDM+FDM**：IDM 压缩相邻帧为 $d_\ell=4$ 维潜在动作；FDM 分解为运输算子 $K_t$ + 残差 $\delta_t$
3. **语义空间动作**：动作 token 建模语义状态迁移，而非像素变化（区别于 LAPA）
4. CFG scale=1.0 即可达到最优：语言对齐更强，无需无条件分支

## 代表工作

- [[RepWAM]]: RepViTok 是核心组件，替换 WAN2.2 VAE 后 RoboTwin Easy 成功率从 78.0 提升至 86.6

## 相关概念

- [[Visual Tokenizer]]
- [[Latent Action]]
- [[Feature Alignment]]
- [[Inverse Dynamics Model]]
- [[Forward Dynamics Model]]
- [[World-Action Model]]
