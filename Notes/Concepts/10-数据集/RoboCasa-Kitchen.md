---
type: concept
aliases: [RoboCasa, RoboCasa Kitchen, RoboCasa365]
---

# RoboCasa-Kitchen

## 定义

RoboCasa-Kitchen 是基于 RoboCasa 仿真平台的厨房场景机器人操控 benchmark，包含多种厨房物品操控任务（开关门、抽屉操作、微波炉、锅具等），用于评测机器人策略的泛化能力。

## 核心要点

1. **24 类厨房任务**: 涵盖门操控、抽屉、家电、锅具等日常厨房操作
2. **高多样性**: 物体外观、场景布置随机化，测试泛化能力
3. **RoboCasa365 子集**: 用于预训练的 365 种变体场景数据（约 78K 轨迹）
4. **典型任务**: CloseSingleDoor（100%）、CloseDrawer（99.3%）、PnPMicrowaveToCounter（11.3%）

## 代表工作

- [[GAM]]: 在 RoboCasa-Kitchen 上达到 69.4% 平均成功率，超过 Cosmos-Policy（67.1%）

## 相关概念

- [[LIBERO]]
- [[World-Action Model]]
- [[VLA]]
