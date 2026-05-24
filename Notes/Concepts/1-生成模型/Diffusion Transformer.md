---
type: concept
aliases: [DiT, 扩散变换器, Diffusion Transformer]
---

# Diffusion Transformer

## 定义

Diffusion Transformer（DiT）是将 [[Transformer]] 架构与扩散模型（[[扩散模型|Diffusion Model]]）相结合的生成网络，以 Transformer 的自注意力机制替代传统扩散模型中的 U-Net 主干，实现高质量的生成任务。

## 数学形式

在条件流匹配框架下，DiT 的核心推理形式为：

$$
\hat{\mathbf{v}}^{\tau} = \mathrm{DiT}_{\theta}(c, \mathbf{x}^{\tau}, \tau)
$$

其中 $c$ 为条件输入（语言/视觉特征），$\mathbf{x}^{\tau}$ 为当前噪声状态，$\tau$ 为去噪时间步，$\hat{\mathbf{v}}^{\tau}$ 为预测的速度场。

## 核心要点

1. **架构替换**: 以 Transformer 块（自注意力 + 前馈网络）替代 U-Net，支持更灵活的全局建模
2. **条件注入**: 通过 AdaLN（Adaptive Layer Norm）或 cross-attention 将条件信息注入每个 Transformer 层
3. **扩展性强**: 参数规模和计算量可随层数线性扩展，支持从小模型到大模型的统一框架
4. **应用广泛**: 原始用于图像生成（如 DiT、SD3），后扩展至视频生成、机器人动作生成等领域

## 代表工作

- [[RoVLA]]: 使用 32 层 DiT 在条件流匹配框架下生成机器人动作块
- Peebles & Xie (2023), "Scalable Diffusion Models with Transformers": 原始 DiT 论文，图像生成

## 相关概念

- [[Flow Matching]]: DiT 常与流匹配框架结合用于连续生成
- [[Conditional Flow Matching]]: 条件流匹配，DiT 的常用训练范式
- [[Action Chunking]]: 机器人策略中 DiT 的输出形式
- [[扩散模型]]: DiT 所属的生成模型大类
