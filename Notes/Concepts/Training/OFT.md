---
type: concept
aliases: [OFT, Orthogonal Fine-Tuning, 正交微调]
---

# OFT (Orthogonal Fine-Tuning)

## 定义
OFT 是一种参数高效微调（PEFT）方法，通过对权重矩阵施加正交变换来保留预训练模型的超球面能量（hyperspherical energy），从而在微调过程中保持神经元之间的相对角度关系不变。

## 数学形式
$$W' = RW, \quad R^T R = I$$

其中 $R$ 是可学习的正交矩阵，$W$ 是预训练权重。

## 核心要点
1. 正交变换保持向量模长，防止表示崩塌（representation collapse）
2. 在 VLA 中，OFT 能在 fine-tune 时保留 VLM 预训练表示的几何结构
3. 相比 LoRA：OFT 约束更强，对 VLA 初始化可能更友好（见 VLA-Repr 研究）
4. 可作为 LoRA 的替代选项，适用于需要严格保留 backbone 表示的场景

## 与 LoRA 对比
- [[LoRA]]: 低秩分解，改变权重方向；OFT: 正交旋转，保持神经元关系
- VLA 场景下，OFT 在某些任务上优于 LoRA（摘要中出现在多篇 VLA 论文的 baseline 中）

## 代表工作
- Qiu et al. (2023): OFT 原始论文
- [[OpenVLA-OFT]]: 将 OFT 应用于 OpenVLA 训练
- [[VLA-Repr]]: 比较 OFT 与 LoRA 在 VLA 初始化中的作用

## 相关概念
- [[LoRA]]
- [[正交正则化]]
- [[Representation Collapse]]
