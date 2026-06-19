---
type: concept
aliases: [Prompting with Iterative Visual Optimization]
---

# PIVOT

## 定义
Prompting with Iterative Visual Optimization，通过在图像上叠加视觉提示（候选动作的视觉标注）并用 VLM 迭代选择最优提示来驱动机器人策略，无需额外训练。

## 核心要点
1. Training-free：直接使用冻结 VLM 的视觉理解能力
2. 将动作空间转化为 VLM 可评估的视觉选项
3. 迭代细化：多轮提示缩小候选范围直到收敛

## 代表工作
- Nasiriany et al. (2024): PIVOT 原始论文
- [[VAP]]: 对比基线，用更高效的 grounding 替代迭代优化
- [[SlowFastNav]]: 类似 VLM-driven 选择机制

## 相关概念
- [[VAP]]
- [[VLA]]
- [[MOKA]]
