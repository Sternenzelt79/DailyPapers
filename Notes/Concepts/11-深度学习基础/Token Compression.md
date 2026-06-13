---
type: concept
aliases: [Token 压缩, 视觉Token压缩, Sparse Tokens]
---

# Token Compression

## 定义
通过减少 Transformer 处理的 Token 数量来降低计算复杂度（二次方复杂度与序列长度相关），常见于视觉 Transformer 的效率优化。

## 核心要点
1. 方法包括：降低输入分辨率、Token 剪枝、Token 合并（ToMe）等
2. 视觉场景中，高空间冗余使得适度压缩对任务性能影响有限
3. Efficient-WAM 中通过降低未来帧分辨率（384×320→192×160）实现 75% Token 压缩（240→60 tokens）

## 代表工作
- [[Efficient-WAM]]: 多尺度视频潜变量布局，当前帧高分辨率、未来帧低分辨率

## 相关概念
- [[Transformer]]
- [[Video Diffusion Model]]
