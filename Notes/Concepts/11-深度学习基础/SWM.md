---
type: concept
aliases: [Stable World Model, stable-worldmodel]
---

# SWM

## 定义
Stable World Model（SWM）：由 Mila/NYU 团队开发的模块化、可复现的世界模型研究生态系统，提供标准化环境、数据收集工具、规划算法和基准实现，旨在减少研究者的"想法到实验"时间差。

## 数学形式
SWM 围绕统一 `World` 接口设计：
$$
o_{t+1} = \text{World.step}(o_t, a_t)
$$
接口标准化后，训练好的动力学模型可以直接插入 MPC 规划循环。

## 核心要点
1. **模块化设计**：数据收集、环境、规划算法（[[MPC]]、MPPI 等）相互解耦，可独立替换
2. **标准化环境**：包含 Push-T、TwoRoom 等多种操控任务，支持可控变异因子（视觉、物理属性）
3. **基准实现**：包含 [[DINO]]-WM 等主流 world model 的参考实现，便于复现
4. **零样本鲁棒性测试**：内置 Factor-of-Variation（FoV）机制，系统测试分布外泛化

## 代表工作
- [[stable-worldmodel]]: 提出 SWM 的原始论文（Maes et al., 2026）

## 相关概念
- [[MPC]]
- [[PLDM]]
- [[PushT]]
