---
type: concept
aliases: [SIMPLER benchmark, Scalable Imitation Learning Evaluation]
---

# SIMPLER

## 定义

面向机器人操控策略的跨环境泛化评测基准，通过 Visual Matching 和 Variant Aggregation 两种设置，测试策略在不同视觉环境条件下的鲁棒性。

## 核心要点

1. **Visual Matching**: 仿真环境视觉外观与训练分布接近的设置
2. **Variant Aggregation**: 汇总多种视觉变体（光照、背景等）的整体性能
3. **典型任务**: Move Near、Pick Coke Can、Open/Close Drawer

## 代表工作

- [[VLA-Pruner]]: 在 SIMPLER 的 75% 剪枝率设置下评测，VLA-Pruner 取得 96.8% 相对精度，远超 FastV（73.1%）

## 相关概念

- [[VLA]]
- [[LIBERO]]
