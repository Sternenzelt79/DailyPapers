---
type: concept
aliases: [DreamZero, Dream Zero]
---

# DreamZero

## 定义

DreamZero 是首先对 World-Action Model（WAM）范式进行形式化定义的工作，将世界模型与动作模型的耦合训练统一到一个框架下。

## 核心要点

1. **范式形式化**：明确定义了 World-Action Model 的组成——世界模型（生成想象未来）+ 动作模型（翻译为可执行动作）
2. **生成式规划**：在隐空间中通过世界模型进行规划，再由动作模型解码
3. **基础性工作**：为后续 AdaWAM、WAM-RL 等工作建立了统一的研究范式

## 代表工作

- [[WAM-RL]]: 在 DreamZero 范式基础上引入强化学习联合优化

## 相关概念

- [[World-Action Model]]
- [[World Model]]
- [[WAM-RL]]
- [[Genie Envisioner]]
