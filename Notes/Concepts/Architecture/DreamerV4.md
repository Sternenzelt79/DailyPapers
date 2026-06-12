---
type: concept
aliases: [DreamerV4, Dreamer v4]
---

# DreamerV4

## 定义
DeepMind 的 Dreamer 系列世界模型的第四代，基于 RSSM（Recurrent State Space Model）从头学习紧凑潜空间表示，通过在潜空间中的"梦境"轨迹训练策略，减少真实环境交互需求。

## 核心要点
1. **从头训练编码器**: 不依赖预训练视觉特征，在目标域内端到端学习表示
2. **RSSM 架构**: 结合确定性和随机性状态，建模环境随机性
3. **潜空间 Actor-Critic**: 策略和价值网络完全在梦境潜空间中训练
4. **局限性**: 从零训练编码器导致分布外（OOD）鲁棒性较差，依赖大量目标域数据

## 代表工作
- Hafner et al., "Mastering Diverse Domains through World Models"

## 相关概念
- [[Flow Matching]]
- [[VAE]]
- [[Critic Network]]
- [[WEAVER]]: 通过使用预训练编码器克服了 DreamerV4 的 OOD 局限
