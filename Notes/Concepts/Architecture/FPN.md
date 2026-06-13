---
type: concept
aliases: [Feature Pyramid Network, 特征金字塔网络, FPN]
---

# FPN（Feature Pyramid Network）

## 定义

FPN 是一种多尺度特征提取架构，通过**自顶向下路径**和**横向连接**将深层语义特征与浅层空间特征融合，生成多分辨率的统一特征图。

## 数学形式

$$
P_k = \text{Conv}_{1\times1}(C_k) + \text{Upsample}(P_{k+1})
$$

$$
P_k = \text{Conv}_{3\times3}(P_k)
$$

自底向上提取多尺度特征 $\{C_2, C_3, C_4, C_5\}$，自顶向下融合得到 $\{P_2, P_3, P_4, P_5\}$。

## 核心要点

1. **多尺度融合**：浅层保留空间细节，深层提供语义信息；
2. **参数轻量**：仅用 $1\times1$ 和 $3\times3$ 卷积，计算高效；
3. **在视频/世界模型中的应用**：从 U-Net 解码器各层提取多尺度潜特征并聚合，如 MemoryVLA++ 的想象模块。

## 代表工作

- [[MemoryVLApp]]: 用 FPN 聚合世界模型 U-Net 多尺度特征得到 $z \in \mathbb{R}^{K \times N_z \times d_p}$

## 相关概念

- [[世界模型]]
- [[Stable Video Diffusion]]
- [[Cross-Attention]]
