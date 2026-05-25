---
type: concept
aliases: [CoTracker3, Co-Tracker]
---

# CoTracker3

## 定义
Meta AI 提出的视频点追踪方法，使用 Transformer 架构对视频中任意点进行长时序一致追踪；CoTracker3 是第三代，引入半监督训练策略，显著提升真实视频上的追踪精度。

## 数学形式
$$\hat{p}_{1:T} = \text{CoTracker}(I_{1:T}, p_0)$$

追踪从初始点 $p_0$ 出发，在 $T$ 帧视频 $I_{1:T}$ 中预测轨迹。

## 核心要点
1. 基于 Transformer，支持多点同时追踪（联合注意力建模点间关系）
2. 半监督训练：用真实视频的伪标签提升对真实场景的泛化
3. 引入 sliding window 策略处理长视频
4. 在机器人视频世界模型中用于提供稠密运动监督信号

## 代表工作
- [[GEM-4D]]: 用 CoTracker3 追踪提取 4D 对应关系，注入视频世界模型的几何监督

## 相关概念
- [[JOPAT]]
- [[VGGT]]
- [[扩散世界模型]]
