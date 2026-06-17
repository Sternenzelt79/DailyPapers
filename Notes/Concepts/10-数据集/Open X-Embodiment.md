---
type: concept
aliases: [Open X-Embodiment Dataset, OXE, Open-X]
---

# Open X-Embodiment

## 定义

Open X-Embodiment 是由多个机器人研究机构联合构建的大规模跨机器人平台数据集，包含来自多种机器人、多种任务和多种环境的操控轨迹，旨在推动通用机器人策略的预训练。

## 核心要点

1. **跨平台多样性**: 涵盖多种机器人形态（单臂、双臂等）和操控任务
2. **大规模**: 数十万条机器人演示轨迹
3. **社区协作**: 由 Google、斯坦福、UC Berkeley 等多个机构联合贡献
4. **标准格式**: 统一的 RLDS 数据格式，便于多数据集混合训练

## 代表工作

- [[GAM]]: 使用 Open X-Embodiment 作为预训练主要数据源（72%，约 563K 轨迹）
- RT-2、π0 等 VLA 均基于此数据集预训练

## 相关概念

- [[VLA]]
- [[Imitation Learning]]
- [[World-Action Model]]
