---
type: concept
aliases: [Video Memory, surfel-indexed video memory]
---

# VMem

## 定义

VMem 是一种基于 surfel 索引的视频记忆方法，将历史视频帧与静态 3D surfel 关联，用于在长视频生成中检索几何相关的历史帧；是 [[Mem-World]] 中 W-VMem 的前驱工作。

## 核心要点

1. **Surfel 索引**：将历史帧观测锚定到 3D 场景表面的 surfel 上，实现几何感知的历史帧检索
2. **静态场景假设**：原始 VMem 面向静态场景，surfel 缺乏时间感知（无时间戳）和任务感知（无操作对象标志）
3. **与 W-VMem 的区别**：[[Mem-World]] 的 W-VMem 将其扩展为 4D（加入时间戳 $t_k$ 和操作对象标志 $m_k$），并专为腕部视角和机器人操作动态场景设计

## 代表工作

- [[Mem-World]]: 将 VMem 扩展为动态感知的 4D W-VMem，应用于机器人操作世界模型

## 相关概念

- [[Surfel Map]]
- [[Action-Conditioned Video Generation]]
- [[Video Diffusion Model]]
