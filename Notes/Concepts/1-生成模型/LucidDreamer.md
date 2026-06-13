---
type: concept
aliases: [LucidDreamer]
---

# LucidDreamer

## 定义
基于扩散模型的沉浸式 3D 场景生成系统，通过迭代式 point cloud 扩展和 inpainting 从单张图像生成可漫游的 3D 世界。

## 核心要点
1. 用 RGB point cloud 作为 3D 记忆，每步生成时投影、渲染、inpaint
2. 生成质量高但计算成本较大（需反复 VAE 编码）
3. 是 Latent Spatial Memory 等工作要解决的效率瓶颈的代表

## 代表工作
- 作为视频/场景 WM 的对比 baseline

## 相关概念
- [[WonderWorld]]
- [[FlashWorld]]
- [[ViewCrafter]]
- [[DepthAnything]]
