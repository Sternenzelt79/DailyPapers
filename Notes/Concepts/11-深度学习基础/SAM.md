---
type: concept
aliases: [Segment Anything Model, SAM, 分割一切]
---

# SAM

## 定义
Meta AI 发布的通用图像分割基础模型，通过点、框、文本等多种提示（prompt）对任意物体进行零样本分割。

## 数学形式
$$\text{mask} = \text{SAM}(\text{image}, \text{prompt})$$

提示可为点坐标、边框或文字；输出为二值分割掩码及置信分数。

## 核心要点
1. 在 SA-1B（10亿+掩码）上预训练，泛化能力极强
2. 支持交互式分割：点击一个点即可分割对应物体
3. 零样本泛化到新类别和新场景，无需额外微调
4. 在 3DGS 中用于提供物体级语义监督（见 [[OP2GS]]）
5. 对细小、透明、高度遮挡物体分割效果有限

## 代表工作
- Kirillov et al. (2023): Segment Anything (SAM)
- SAM 2 (2024): 扩展到视频分割

## 相关概念
- [[LERF]]
- [[3D Gaussian Splatting]]
- [[DINO]]
