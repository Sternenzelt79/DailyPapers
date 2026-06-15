---
type: concept
aliases: [AstriBot, S1机器人]
---

# AstriBot S1

## 定义

AstriBot 公司开发的双臂人形机器人平台，具备高精度末端执行器控制能力，适用于接触密集型操作任务的研究与部署。

## 核心要点

1. **双臂配置**: 支持双臂协调操作，动作空间为 16 维（每臂：3D 位置 + 四元数 + 1 维夹爪）
2. **真实世界评测**: 作为 WAM4D、LingBot-VA、Fast-WAM 等世界动作模型的真实机器人评测平台
3. **任务类型**: 适用于盘子取放、瓶子摆放、LEGO 积木分拣、钢笔装帽等精密操作任务

## 代表工作

- [[WAM4D]]: 在 AstriBot S1 上完成真实世界实验，4 类任务平均子动作成功率 0.90
- [[LingBot]]: AstriBot S1 的主要策略架构之一

## 相关概念

- [[World Action Model]]
- [[RoboTwin 2.0]]
