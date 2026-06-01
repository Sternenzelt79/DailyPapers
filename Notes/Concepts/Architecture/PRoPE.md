---
type: concept
aliases: [PRoPE, Projectivity-aware Rotary Position Encoding, 射影感知旋转位置编码]
---

# PRoPE

## 定义
Projectivity-aware Rotary Position Encoding：一种将相机内外参融合进旋转位置编码的相机注入方法，通过射影矩阵扩展标准 RoPE，使视频 transformer 的 self-attention 具备几何感知的相机控制能力。

## 数学形式

提升射影矩阵：

$$
\tilde{P}_i = \begin{bmatrix} K_i & 0 \\ (T_i^{cw})^\top & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 4}
$$

块对角旋转矩阵：

$$
D_t^{PRoPE} = \begin{bmatrix} I_{a/8} \otimes \tilde{P}_{i(t)} & 0 \\ 0 & \begin{bmatrix} RoPE_{a/4}(x_t) & 0 \\ 0 & RoPE_{a/4}(y_t) \end{bmatrix} \end{bmatrix}
$$

PRoPE Attention：

$$
\text{Attn}^{PRoPE}(Q,K,V) = D^{PRoPE} \odot \text{Attn}\left((D^{PRoPE})^\top \odot Q,\; (D^{PRoPE})^{-1} \odot K,\; (D^{PRoPE})^{-1} \odot V\right)
$$

## 核心要点
1. 将相机内参 $K$ 和外参 $T^{cw}$ 融合为 4×4 射影矩阵，统一编码视角信息
2. 以乘法方式注入 attention，利用射影变换的群结构保持几何一致性
3. head 维度中 1/8 用于相机射影编码，1/4 用于标准 2D 坐标 RoPE
4. 无需额外 cross-attention 模块，对原始架构侵入性小

## 代表工作
- [[minWM]]: 提出 PRoPE 并用于将 T2V/TI2V 基础模型转化为相机可控世界模型

## 相关概念
- [[RoPE]]: 旋转位置编码，PRoPE 的技术基础
- [[射影变换]]: 相机矩阵的数学基础
