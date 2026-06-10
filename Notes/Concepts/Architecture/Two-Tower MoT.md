---
type: concept
aliases: [Two-Tower Mixture-of-Transformers, 双塔混合变换器, Two-Tower Architecture]
---

# Two-Tower MoT

## 定义

一种将自回归变换器（AR Tower）和扩散变换器（DM Tower）融合在同一前向传播中的混合架构，两塔共享 joint attention，分别处理离散 token（推理）和连续 token（生成），由 Cosmos 3 提出。

## 数学形式

Token 序列被划分为两个子序列：

$$
\mathbf{X} = [\underbrace{x_1^{AR}, \ldots, x_m^{AR}}_{\text{AR 子序列（离散）}}, \underbrace{x_1^{DM}, \ldots, x_n^{DM}}_{\text{DM 子序列（连续）}}]
$$

AR 和 DM token 使用独立的 LayerNorm 和 MLP，但共享同一 Attention 算子（Joint Attention）。

## 核心要点

1. **AR Tower（Reasoner）**: 因果自注意力，处理文本和 ViT 视觉 token，用于理解和推理
2. **DM Tower（Generator）**: 全注意力，处理 VAE 连续 latent，用于图像/视频/音频/动作生成
3. **Joint Attention**: AR 和 DM 子序列互相可见，推理理解直接条件化生成
4. **独立参数**: 两塔在 MLP 和 LayerNorm 层使用独立参数（MoT 的"专家"设计），在 Attention 层共享
5. **双模式**: Reasoner 模式仅激活 AR 塔；Generator 模式同时激活两塔

## 代表工作

- [[Cosmos3]]: 提出 Two-Tower MoT，首个将 VLM + 视频生成 + 世界仿真 + 策略模型统一的 omnimodal 框架

## 相关概念

- [[Mixture-of-Transformers]]
- [[Diffusion Transformer]]
- [[自回归变换器]]
- [[多维旋转位置编码]]
