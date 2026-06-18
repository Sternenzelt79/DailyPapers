---
type: concept
aliases: [Le World Model, 确定性JEPA世界模型]
---

# LeWorldModel

## 定义
LeWorldModel 是一种确定性的联合嵌入预测架构（JEPA）风格的世界模型，使用 SigReg 正则化训练潜在表示，用于机器人策略的辅助预测模块。

## 核心要点
1. 确定性动力学：无显式随机潜在变量，通过预测器输出确定性下一状态
2. SigReg 正则化：防止潜在空间坍缩
3. 消融结果：确定性表示在某些任务上足以胜任，有时甚至优于随机的 RSSM 方法

## 代表工作
- [[DREAM-Chunk]]: 消融实验中作为确定性世界模型选项，与 R2-Dreamer 和 EB-JEPA 对比

## 相关概念
- [[JEPA]]
- [[潜在世界模型]]
- [[R2-Dreamer]]
- [[EB-JEPA]]
