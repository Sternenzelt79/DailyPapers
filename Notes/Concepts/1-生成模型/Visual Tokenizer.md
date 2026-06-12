---
type: concept
aliases: [视觉 Tokenizer, Video Tokenizer, ViT Tokenizer, VAE Tokenizer]
---

# Visual Tokenizer

## 定义

将原始视频/图像帧压缩为紧凑潜在 token 的编码-解码器模块；在世界模型和视频生成领域，tokenizer 的设计直接决定下游模型能学到的语义质量。

## 数学形式

标准重建目标：

$$
\mathcal{L}_{rec} = \lambda_1 \|o - \hat{o}\|_1 + \lambda_{perc}\mathcal{L}_{perc}(o, \hat{o}) + \lambda_{gan}\mathcal{L}_{gan}(\hat{o})
$$

## 核心要点

1. **重建导向**（如 WAN VAE）：优化像素保真度，语义信息稀薄
2. **语义导向**（如 RepViTok）：加入特征对齐损失，使 token 继承基础模型语义
3. Spatio-temporal patching：图像 $16\times16$，视频帧 $4\times16\times16$ tubelet
4. 解码器通过转置卷积重建像素

## 代表工作

- [[RepWAM]]: 提出 RepViTok，在重建目标外加入特征对齐，语义对齐后 token 更易用于机器人控制

## 相关概念

- [[Causal 3D VAE]]
- [[Feature Alignment]]
- [[World-Action Model]]
- [[RepViTok]]
