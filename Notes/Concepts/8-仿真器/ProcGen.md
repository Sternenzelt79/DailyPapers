---
type: concept
aliases: [ProcGen, Procedural Generation Benchmark]
---

# ProcGen

## 定义
ProcGen（Procedural Generation Benchmark）是 OpenAI 发布的 16 个程序化生成 2D 游戏环境的强化学习 benchmark，用于测试泛化能力。

## 核心要点
1. 每个环境有无限多程序化生成的关卡，测试泛化而非记忆
2. 训练/测试关卡完全不同，强迫算法学到真正可泛化策略
3. 广泛用于 world model、model-based RL 方法评测

## 代表工作
- Cobbe et al. (2020), Leveraging Procedural Generation to Benchmark RL

## 相关概念
- [[CLAW]]
- [[DreamerV3]]
