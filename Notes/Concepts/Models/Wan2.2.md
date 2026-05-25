---
type: concept
aliases: [Wan 2.2, Wan2.2 TI2V, Wan Video]
---

# Wan2.2

## 定义
Wan2.2：阿里巴巴推出的开源大规模视频生成模型，支持文本到视频（T2V）和文本+图像到视频（TI2V）等多种生成模式，5B 参数规模，在多项视频生成基准上达到 SOTA 水平。

## 数学形式
密集视频生成：给定初始帧 $x_0$ 和文本条件 $l$，生成包含 $N$ 帧的视频序列：
$$V = \{x_0, x_1, \ldots, x_{N-1}\} = G_\theta(x_0, l)$$

## 核心要点
1. **TI2V 模式**: Text-Image-to-Video，以初始帧+文本为条件生成后续视频帧，适用于机器人轨迹预测等应用
2. **密集视频输出**: 生成完整的 81 帧视频，包含大量冗余中间帧
3. **推理效率**: 在 NVIDIA H20 GPU 上生成 81 帧约需 400 秒，比稀疏关键帧方法慢 40x 以上
4. **机器人应用局限**: 在 SWEET 对比中，Wan2.2 更容易出现 Implausible Grasp、Object Deformation、Arm Deformation 和 Object Teleportation 等规划级失败

## 代表工作
- [[SWEET]]: 以 Wan2.2 TI2V 5B 作为视频生成基线进行对比，关键帧质量和推理效率均低于 FLUX-Kontext

## 相关概念
- [[FLUX-Kontext]]: 图像编辑对比方法，稀疏关键帧预测更高效
- [[World Model]]: Wan2.2 可作为视觉世界模型用于机器人规划
- [[Video Diffusion]]: 视频生成的技术范式
