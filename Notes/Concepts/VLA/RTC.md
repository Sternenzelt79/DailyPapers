---
type: concept
aliases: [Real-Time Action Chunking, 实时动作分块]
---

# RTC (Real-Time Action Chunking)

## 定义
RTC 是一种动作分块改进方法，通过 inpainting 技术将新采样的动作块与已执行动作对齐，解决块间衔接的不连续问题。

## 核心要点
1. 使用 inpainting 约束：新块的起始动作应与历史执行动作连续
2. 关注**块间衔接**（cross-chunk transition）的平滑性
3. 与 BID 类似，侧重块边界而非块内响应性

## 代表工作
- [[DREAM-Chunk]]: 对比方法，DREAM-Chunk 在块内随机扰动下优于 RTC

## 相关概念
- [[Action Chunking]]
- [[BID]]
- [[测试时扩展]]
