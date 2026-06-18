---
type: concept
aliases: [Adaptive Layer Norm Zero, AdaLN-Zero 初始化]
---

# AdaLN-Zero

## 定义

AdaLN-Zero 是 DiT（Diffusion Transformer）中的一种条件归一化策略：将调制残差分支的输出门控参数（gate）初始化为零，使模型在训练初始阶段等同于原预训练网络，从而保护预训练权重并稳定微调过程。

## 数学形式

$$
\hat{Z} = Z + \underbrace{\text{gate}}_{\text{初始}=0} \cdot f(Z)
$$

其中 gate 由 AdaLN 的线性层产生，初始化使其输出为 0。

## 核心要点

1. **零初始化保护**: 新增模块在训练开始时输出为零，不破坏预训练模型的表示
2. **渐进激活**: 随训练推进，gate 逐渐学习到非零值，新模块逐步发挥作用
3. **广泛应用**: 在向预训练 DiT 中插入新模块时（如跨视图注意力、ControlNet）常用此策略

## 代表工作

- [[Diffusion Transformer]]: DiT 原始论文中提出 AdaLN-Zero
- [[PAIWorld]]: 在插入跨视图注意力块时使用 AdaLN-Zero 保护 Cosmos-Predict2.5 预训练权重

## 相关概念

- [[Diffusion Transformer]]
- [[Cross-View Attention]]
- [[RoPE]]
