---
type: concept
aliases: [Open X-Embodiment, OXE, Open-X Embodiment]
---

# OXE（Open X-Embodiment）

## 定义
Open X-Embodiment（OXE）是一个跨机器人、跨任务的大规模遥操作数据聚合数据集，统一了来自 22 种机器人的 100 万以上条轨迹，旨在支持跨形态的泛化策略学习。

## 数学形式
每条轨迹为 $(o_1, a_1, o_2, a_2, \ldots, o_T, a_T)$ 的状态-动作序列，观测模态包括 RGB、深度、点云、语言指令等。

## 核心要点
1. **规模**: 1M+ 轨迹，160K+ 任务，60 种场景，22 种机器人
2. **跨形态泛化**: 强制世界模型将通用物理规律与特定机器人运动学解耦，是 WAM 训练的重要基础
3. **数据来源混合**: 遥操作 + 脚本 + 人工演示

## 代表工作
- [[WAM-Survey]]: 将 OXE 列为 WAM 训练的核心遥操作数据集
- RT-2：利用 OXE 数据训练的代表性 VLA

## 相关概念
- [[VLA]]: OXE 是 VLA 和 WAM 训练的核心数据资源
- [[WAM]]: WAM 可利用 OXE 中的 $(o, a, o')$ 三元组进行联合训练
