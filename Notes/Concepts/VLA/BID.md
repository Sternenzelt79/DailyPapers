---
type: concept
aliases: [Bidirectional Decoding, 双向解码]
---

# BID (Bidirectional Decoding)

## 定义
BID 是一种测试时扩展方法，通过评估多个候选动作块之间的双向一致性来选择最优动作块，提升 VLA 策略在跨块边界处的连贯性。

## 核心要点
1. 采样多个候选动作块，评估新块与历史执行轨迹的一致性
2. 关注**块间一致性**（cross-chunk consistency），与 DREAM-Chunk 的块内反应性互补
3. 无需重新训练策略，属于 policy-agnostic 的测试时方法

## 代表工作
- [[DREAM-Chunk]]: 对比方法，DREAM-Chunk 在随机动力学下优于 BID

## 相关概念
- [[Action Chunking]]
- [[RTC]]
- [[测试时扩展]]
