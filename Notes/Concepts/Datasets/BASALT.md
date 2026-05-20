---
type: concept
aliases: [BASALT benchmark, MineRL BASALT]
---

# BASALT

## 定义
基于 Minecraft 的人类反馈强化学习基准，包含四个无明确奖励函数的开放式任务，需要学习人类偏好。

## 核心要点
1. 四个任务：FindCave、MakeWaterfall、CreateVillageAnimalPen、BuildVillageHouse
2. 没有手工设计的奖励函数，评估依赖人类评判者
3. 用于测试从人类反馈/演示学习复杂行为序列的算法
4. 与 [[VPT]] 配合：VPT 提供基础 policy，BASALT 微调用于特定任务
5. 在 world model 研究中用作稀有转换课程学习的测试床（见 [[PROWL]]）

## 代表工作
- Shah et al. (2022): BASALT: A Benchmark for Learning from Human Feedback

## 相关概念
- [[VPT]]
- [[RLHF]]
- [[Reinforcement Learning]]
