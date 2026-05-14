---
type: concept
aliases: [Diffusion Transformer, 扩散变换器]
---

# DiT

## 定义
Diffusion Transformer：用 Transformer 替代 UNet 作为扩散模型骨干，通过 Adaptive Layer Norm 注入时间步和类别条件，在图像/视频生成中表现 SOTA。

## 数学形式
$$\text{adaLN-Zero: } [\gamma, \beta] = \text{MLP}(c), \quad y = \gamma \cdot \text{LN}(x) + \beta$$

## 核心要点
1. 将图像/视频 patch 化为 token 序列，送入标准 Transformer blocks
2. 时间步 t 和类别标签 c 通过 [[AdaLN]] 注入，不用 cross-attention
3. 相比 UNet 扩展性更强，参数量越大效果越好（遵循 scaling law）
4. 是 [[FLUX]]、[[SD3]]、Sora 等现代生成模型的核心架构

## 代表工作
- Peebles & Xiao 2023，Scalable Diffusion Models with Transformers
- [[MM-DiT]]: FLUX/SD3 使用的双流 DiT 变体

## 相关概念
- [[DDPM]]
- [[AdaLN]]
- [[MM-DiT]]
- [[LDM]]
