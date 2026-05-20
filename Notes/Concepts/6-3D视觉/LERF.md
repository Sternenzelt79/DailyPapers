---
type: concept
aliases: [Language Embedded Radiance Fields, 语言嵌入辐射场]
---

# LERF

## 定义
将 CLIP 语言特征嵌入 NeRF 辐射场的方法，实现开放词汇的 3D 场景语义查询，无需预定义类别。

## 数学形式
$$\phi(\mathbf{x}) = \text{CLIP-embed}(\text{patch around } \mathbf{x})$$

$$\text{relevance}(\mathbf{x}, q) = \frac{\exp(\phi(\mathbf{x}) \cdot \text{CLIP-text}(q))}{\sum_{c} \exp(\phi(\mathbf{x}) \cdot \text{CLIP-text}(c))}$$

## 核心要点
1. 在 NeRF 训练过程中联合学习几何和 CLIP 语义特征
2. 支持任意自然语言 query 在 3D 空间中定位物体
3. 相比 LERF，[[3D Gaussian Splatting]] 版本（如 [[OP2GS]]）计算更高效
4. 对模糊语言描述和强遮挡场景的定位精度有限

## 代表工作
- Kerr et al. (2023): LERF: Language Embedded Radiance Fields

## 相关概念
- [[3D Gaussian Splatting]]
- [[SAM]]
- [[CLIP]]
