---
type: concept
aliases: [SSIM, Structural Similarity Index, 结构相似度]
---

# SSIM（结构相似度）

## 定义

SSIM（Structural Similarity Index）是衡量图像感知质量的指标，通过比较亮度、对比度和结构三个维度来评估两幅图像的相似性，比 PSNR 更接近人类视觉感知。

## 数学形式

$$
\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

其中：
- $\mu_x, \mu_y$：图像 $x$, $y$ 的均值（亮度）
- $\sigma_x^2, \sigma_y^2$：方差（对比度）
- $\sigma_{xy}$：协方差（结构）
- $C_1, C_2$：防止分母为零的稳定常数

## 核心要点

1. **范围**：$[-1, 1]$，值越接近 1 表示越相似
2. **三维分解**：亮度（luminance）+ 对比度（contrast）+ 结构（structure）
3. **比 PSNR 更接近感知**：考虑了图像的局部统计特性和结构信息
4. **WAM 评估应用**：与 PSNR 联合作为世界模型帧重建的基础质量指标

## 代表工作

- [[WAMSurvey]]: WAM 世界建模视觉保真度评估框架

## 相关概念

- [[PSNR]]
- [[LPIPS]]
- [[FVD]]
- [[World Action Model]]
