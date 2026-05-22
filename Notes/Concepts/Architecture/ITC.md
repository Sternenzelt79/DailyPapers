---
type: concept
aliases: [Identifiable Token Correspondence]
---

# ITC

## 定义
Identifiable Token Correspondence，通过显式 token 对应性约束，解决 token-based transformer world model 在长 horizon rollout 中出现的 object duplication / disappearance / transmutation 问题。

## 核心要点
1. 问题：token WM 把 next-frame prediction 当纯生成问题，无跨帧 token identity
2. ITC 添加跨时间步的 token 对应约束
3. 在 MinAtar 等 Atari 环境验证，对比 DreamerV3 / IRIS / STORM

## 代表工作
- [[ITC]]: Kim Youngin et al., 2026 (arXiv 2605.16457)

## 相关概念
- [[DreamerV3]]
- [[World Model]]
