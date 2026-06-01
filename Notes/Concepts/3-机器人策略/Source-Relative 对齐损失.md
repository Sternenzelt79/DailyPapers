---
type: concept
aliases: [SR Loss, Source-Relative Loss, 源相对损失]
---

# Source-Relative 对齐损失（SR Loss）

## 定义

Source-Relative（SR）损失是一种用于跨具身视觉表征对齐的 hinge 损失，以**初始（未适配）特征对之间的距离**作为动态基准，而非固定 margin，鼓励适配后的特征更接近对应的跨具身配对。

## 数学形式

$$
\mathcal{L}_{SR} = \mathbb{E}_{(H,R)\sim\mathcal{D}_p}\left[m_s + d(f^R, f^H) - d(f^{R_0}, f^H)\right]_+
$$

其中 $d(u,v) = 1 - \cos(u,v)$ 为余弦距离，$[\cdot]_+$ 为 hinge（ReLU）函数。

## 核心要点

1. **动态基准**：以初始机器人特征 $f^{R_0}$ 与人类特征的距离为基准，避免固定 margin 设置的问题
2. **Hinge 形式**：只惩罚"还不够好"的配对，不惩罚已经足够对齐的配对
3. **对比方向**：只考虑机器人→人类的对齐（机器人编码器适配），不修改人类编码器

## 代表工作

- [[HARP-VLA]]: 提出 SR 损失用于人机视觉表征对齐，结合 PD 损失组成 SRPD

## 相关概念

- [[Pair-Discriminative 对齐损失]]
- [[余弦距离]]
- [[潜在动作模型]]
- [[跨具身学习]]
