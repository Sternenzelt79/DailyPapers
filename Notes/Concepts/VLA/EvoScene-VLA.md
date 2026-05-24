---
type: concept
aliases: [Evolving Scene Beliefs, Recurrent Scene Prefix]
---

# EvoScene-VLA

## 定义
在 chunked VLA 的 action decoder 内加入循环场景信念（recurrent scene prefix），在 chunk 执行过程中根据已执行动作动态更新场景状态，无需额外视觉输入。

## 核心要点
1. Chunked VLA 每次只用当前帧，EvoScene 在 chunk 内维护 evolving scene 状态
2. 本质是把轻量 world model 内嵌进 action decoder
3. 在 RoboTwin benchmark 测试

## 代表工作
- [[EvoScene-VLA]]: Chushan Zhang et al., 2026 (arXiv 2605.21862)

## 相关概念
- [[MemoryVLA]]
- [[SpatialVLA]]
