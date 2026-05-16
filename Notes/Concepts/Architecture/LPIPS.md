---
type: concept
aliases: [LPIPS, Learned Perceptual Image Patch Similarity, 感知相似度]
---

# LPIPS（学习感知图像块相似度）

## 定义

LPIPS（Learned Perceptual Image Patch Similarity）是在深度特征空间中度量图像感知相似性的指标，通过提取多层深度特征并计算加权距离，比像素级指标更好地捕捉人类感知。

## 数学形式

$$
\text{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \left\| w_l \odot \hat{f}^l(x)_{hw} - \hat{f}^l(y)_{hw} \right\|_2^2
$$

其中：
- $\hat{f}^l(\cdot)_{hw}$：第 $l$ 层位置 $(h,w)$ 处的通道归一化深度特征
- $w_l$：学习到的逐通道权重向量
- $H_l, W_l$：第 $l$ 层特征图的高度和宽度

## 核心要点

1. **基于深度特征**：使用预训练网络（如 VGG、AlexNet）的多层特征进行比较
2. **学习到的权重**：通过人类感知判断数据集训练权重 $w_l$
3. **值越小越好**：与 PSNR/SSIM 相反，LPIPS 值越低表示感知越相似
4. **WAM 评估应用**：在视频生成和世界模型评估中广泛采用，衡量生成帧的感知质量

## 代表工作

- [[WAMSurvey]]: WAM 世界建模视觉保真度评估框架

## 相关概念

- [[PSNR]]
- [[SSIM]]
- [[FVD]]
- [[DreamSim]]
- [[World Action Model]]
