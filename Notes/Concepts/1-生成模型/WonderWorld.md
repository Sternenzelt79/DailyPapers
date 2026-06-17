---
type: concept
aliases: [WonderWorld]
---

# WonderWorld

## 定义
基于图像扩散的交互式 3D 场景生成模型，给定单张图像可以生成可探索的 3D 世界，通过深度估计 + inpainting 迭代扩展视野。

## 核心要点
1. 输入单张图像，输出可漫游的 3D 场景
2. 使用分层深度图（Layered Depth Image）表示场景结构
3. 速度较快（相比 LucidDreamer），但长程一致性有限

## 代表工作
- 作为 Latent Spatial Memory 等视频世界模型的对比 baseline

## 相关概念
- [[LucidDreamer]]
- [[FlashWorld]]
- [[ViewCrafter]]
- [[DepthAnything]]
