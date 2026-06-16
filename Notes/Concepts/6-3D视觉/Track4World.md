---
type: concept
aliases: [Track4World Dataset, Track 4 World]
---

# Track4World

## 定义

Track4World 是用于训练前馈密集 3D 场景追踪的数据集，提供对视频中所有像素的世界坐标系下连续 3D 追踪标注，用于将几何基础模型（GFM）微调以理解动态场景中的时序运动。

## 核心要点

1. **密集 3D 追踪**: 对图像中每个像素提供时序 3D 坐标追踪标注
2. **世界坐标系**: 追踪在世界中心坐标系下进行，而非相机坐标系
3. **前馈推理**: 支持单次前向传播完成追踪，无需迭代优化
4. **GFM 微调用途**: 用于将 Depth Anything 3 微调为能理解时序 3D 运动的 DA3-Giant

## 代表工作

- [[GAM]]: 使用 Track4World 微调后的 DA3-Giant 作为几何骨干网络

## 相关概念

- [[Depth Anything 3]]
- [[时序世界建模]]
- [[6-3D视觉]]
