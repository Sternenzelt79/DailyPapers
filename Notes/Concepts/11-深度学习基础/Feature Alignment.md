---
type: concept
aliases: [特征对齐, Feature Distillation, Semantic Alignment]
---

# Feature Alignment

## 定义

通过损失函数将学生网络（或可训练编码器）的隐变量拉近冻结教师（基础）模型特征的技术，使目标网络在训练中继承教师的语义表征能力。

## 数学形式

$$
\mathcal{L}_{align} = \|avg(W_{align}\, z) - avg(G(o))\|_2^2
$$

其中 $z$ 为目标编码器输出，$G(o)$ 为冻结基础模型特征，$W_{align}$ 为可学习线性投影。

## 核心要点

1. 冻结基础模型参数，仅更新投影矩阵和目标编码器
2. 平均池化（avg）对齐全局语义，避免对位置信息过度依赖
3. 常与重建损失（pixel-level）联合使用，形成语义-保真双目标

## 代表工作

- [[RepWAM]]: 在视觉 tokenizer 中加入特征对齐损失，使视觉隐变量继承基础视觉模型语义，提升操作成功率

## 相关概念

- [[Knowledge Distillation]]
- [[Vision Transformer]]
- [[Inverse Dynamics Model]]
