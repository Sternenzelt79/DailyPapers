---
type: concept
aliases: [R2Dreamer, Recurrent Stochastic State-Space Model World Model]
---

# R2-Dreamer

## 定义
R2-Dreamer 是基于 RSSM（递归随机状态空间模型）的世界模型，结合 Barlow Twins 正则化训练潜在表示，是 DREAM-Chunk 消融实验中表现最优的辅助世界模型架构。

## 网络结构
- **观测编码器**: obs_dim → 512 → 256
- **先验 RSSM head**: hidden size 512，两层
- **后验 head**: hidden size 512，一层
- **确定性动力学**: 256-dim GRU
- **随机状态**: 32×16 → 512 维

## 核心要点
1. 结合确定性 GRU 和随机潜在变量，建模动力学的不确定性
2. Barlow Twins 正则化：通过最大化互相关矩阵对角线、最小化非对角线项防止坍缩
3. 在 DREAM-Chunk 消融中，R2-Dreamer 在高噪声环境下最鲁棒

## 代表工作
- [[DREAM-Chunk]]: 仿真实验中使用 R2-Dreamer 作为默认世界模型配置，效果最优

## 相关概念
- [[RSSM]]
- [[潜在世界模型]]
- [[LeWorldModel]]
- [[EB-JEPA]]
