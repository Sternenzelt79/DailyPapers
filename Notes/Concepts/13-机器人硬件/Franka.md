---
type: concept
aliases: [Franka Emika, Franka Panda, Franka Research 3, FR3]
---

# Franka

## 定义

由德国 Franka Emika 公司（现为 Agile Robots 旗下品牌）开发的 7 自由度轻型协作机械臂，因其高精度、低成本和开放 API 成为机器人学习研究中最广泛使用的平台之一。

## 核心要点

1. **规格**: 7-DOF、最大负载 3kg（FR3 为 3kg）、重复定位精度 ±0.1mm
2. **控制接口**: Franka Control Interface（FCI），支持 1kHz 实时扭矩/位置控制
3. **开放性**: 提供 libfranka C++ API 和 franka_ros，支持 ROS/ROS2
4. **研究应用**: 广泛用于灵巧操作、模仿学习、VLA 策略评估

## 代表工作

- [[LabVLA]]: 在 Franka 平台上验证 sim-to-real 迁移，四类复合任务平均成功率 86.5%

## 相关概念

- [[VLA]]
- [[DiT]]
