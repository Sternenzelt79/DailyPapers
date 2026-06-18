---
type: concept
aliases: [Kinetix Benchmark, Kinetix仿真基准]
---

# Kinetix

## 定义
Kinetix 是一个 2D 物理仿真基准，涵盖运动（locomotion）、操作（manipulation）、Atari 类任务共 12 个环境，常用于测试机器人策略在随机动力学下的鲁棒性。

## 核心要点
1. 2D 物理引擎，任务多样：运动控制、物体操作、游戏类任务
2. 可在动作执行时注入可控 Gaussian 噪声（$\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$），方便系统评估噪声鲁棒性
3. 轻量快速，适合消融实验和大规模对比

## 代表工作
- [[DREAM-Chunk]]: 在 Kinetix 12 个环境上验证了不同噪声水平下的反应性提升

## 相关概念
- [[Action Chunking]]
- [[潜在世界模型]]
