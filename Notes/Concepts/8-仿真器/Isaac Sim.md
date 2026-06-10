---
type: concept
aliases: [NVIDIA Isaac Sim, IsaacSim, Isaac Simulator]
---

# Isaac Sim

## 定义
NVIDIA 基于 Omniverse 平台开发的机器人仿真器，使用 PhysX 物理引擎，支持光线追踪渲染，广泛用于机器人学习的合成数据生成（SDG）。

## 核心要点
1. **PhotoRealistic 渲染**: 基于 RTX 光线追踪，生成接近真实的合成图像
2. **PhysX 物理引擎**: 精确刚体/柔体/流体仿真
3. **合成数据生成（SDG）**: 通过程序化场景生成大规模标注数据
4. **ROS 2 集成**: 原生支持 ROS 2 接口，仿真到实体无缝迁移
5. **IsaacLab 框架**: 高层 RL 训练框架，支持大规模并行仿真

## 代表工作
- [[Cosmos3]]: 使用 Isaac Sim 生成 Physical-Interaction-Scenes 和 Embodied-Robot-Scenes 合成数据集

## 相关概念
- [[扩散世界模型]]
- [[前向动力学模型]]
