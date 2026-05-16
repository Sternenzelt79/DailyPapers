---
type: concept
aliases: [PSNR, Peak Signal-to-Noise Ratio, 峰值信噪比]
---

# PSNR（峰值信噪比）

## 定义

PSNR（Peak Signal-to-Noise Ratio）是衡量图像/视频重建保真度的经典像素级指标，通过最大信号值与均方误差之比的对数比来量化重建质量。

## 数学形式

$$
\text{PSNR}(x, y) = 10 \log \frac{\text{MAX}^2}{\text{MSE}(x, y)}
$$

其中：
- $\text{MAX}$：像素最大值（如 RGB 图像为 255）
- $\text{MSE}(x, y) = \frac{1}{HW} \sum_{h,w} (x_{hw} - y_{hw})^2$：均方误差

## 核心要点

1. **单位为 dB**：值越高表示重建质量越好，通常 >30dB 认为质量较好
2. **局限性**：仅衡量像素级差异，不能很好反映人类感知质量（高 PSNR 图像视觉上可能仍有明显差异）
3. **WAM 评估应用**：作为世界模型视觉保真度评估的基础指标，通常与 [[SSIM]]、[[LPIPS]] 联合使用

## 代表工作

- [[WAMSurvey]]: WAM 世界建模视觉保真度评估框架

## 相关概念

- [[SSIM]]
- [[LPIPS]]
- [[FVD]]
- [[DreamSim]]
- [[World Action Model]]
