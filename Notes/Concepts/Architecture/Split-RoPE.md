---
type: concept
aliases: [分裂RoPE, Split Rotary Position Encoding]
---

# Split-RoPE

## 定义

Split-RoPE 是一种将注意力头按维度拆分为多个子空间，对每个子空间施加不同旋转位置编码的技术，用于同时编码不同粒度或不同类型的位置信息而互不干扰。

## 数学形式

$$
q = [q_1; q_2], \quad \tilde{q} = [\text{RoPE}(q_1, \text{pos}_1);\, \text{RoPE}(q_2, \text{pos}_2)]
$$

在 [[Geometric RoPE]] 中的具体实例：

$$
\tilde{q} = [\underbrace{\text{RoPE}(q_{\text{ray}}, d^v(h,w))}_{\text{像素级射线方向}};\, \underbrace{\text{RoPE}(q_{\text{pose}}, e^v)}_{\text{视图级姿态}}]
$$

## 核心要点

1. **子空间独立编码**: 不同类型的位置信息在各自子空间中独立施加，避免干扰
2. **灵活可扩展**: 子空间数量和维度划分可按需设计（如 $d/2 + d/2$ 或 $d/4 + 3d/4$）
3. **应用场景**: 多模态位置编码、多尺度位置编码、几何感知位置编码

## 代表工作

- [[PAIWorld]]: 使用 Split-RoPE 将头维度拆分为 ray（像素级）和 pose（视图级）两个子空间

## 相关概念

- [[RoPE]]
- [[Geometric RoPE]]
