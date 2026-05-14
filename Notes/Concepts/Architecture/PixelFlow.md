---
type: concept
aliases: [PixelFlow, 像素空间流匹配]
---

# PixelFlow

## 定义
PixelFlow：在原始像素空间（而非隐空间）执行流匹配（Flow Matching）生成的图像/视频生成方法，避免了 VAE 编解码带来的信息损失。

## 数学形式
$$\frac{d x_t}{dt} = v_\theta(x_t, t), \quad x_0 \sim \mathcal{N}(0, I), \quad x_1 \sim p_{data}$$

## 核心要点
1. 直接在像素空间建模，保留更多细节信息（尤其高频纹理）
2. 计算成本高于 LDM，但规避了 VAE 量化/压缩误差
3. L2P 等工作尝试将 LDM 知识迁移到 pixel-space 模型以降低训练成本

## 代表工作
- PixelFlow 系列工作（具体 arXiv 待查）
- 相关：[[LDM]]、[[L2P]]

## 相关概念
- [[LDM]]
- [[Flow Matching]]
- [[DiT]]
