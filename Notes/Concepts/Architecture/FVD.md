---
type: concept
aliases: [FVD, Fréchet Video Distance, 视频 Fréchet 距离]
---

# FVD（Fréchet 视频距离）

## 定义

FVD（Fréchet Video Distance）是衡量视频生成质量的分布级指标，在预训练视频特征空间中计算真实视频与生成视频分布之间的 Fréchet 距离，反映整体逼真度和时序动态。

## 数学形式

$$
\text{FVD} = \| \mu_r - \mu_g \|_2^2 + \text{Tr}\left( \Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2} \right)
$$

其中：
- $\mu_r, \mu_g$：真实/生成视频分布在预训练特征空间中的均值
- $\Sigma_r, \Sigma_g$：对应的协方差矩阵

## 核心要点

1. **分布级比较**：不逐帧对比，而是比较整个数据集级别的分布，捕捉整体逼真度
2. **时序动态感知**：使用预训练视频特征（如 I3D），能反映时序一致性和运动动态
3. **值越低越好**：FVD=0 表示真实视频和生成视频分布完全相同
4. **WAM 评估应用**：是视频生成和世界模型评估中最广泛使用的指标之一

## 与图像级指标的关系

- [[PSNR]]/[[SSIM]]：像素级，单帧对比
- [[LPIPS]]：感知级，单帧对比  
- FVD：分布级，视频整体时序质量

## 代表工作

- [[WAMSurvey]]: WAM 世界建模视觉保真度评估框架

## 相关概念

- [[PSNR]]
- [[SSIM]]
- [[LPIPS]]
- [[DreamSim]]
- [[World Action Model]]
