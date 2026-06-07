---
type: concept
aliases: [MANO Model, Embodied Hand Model, 参数化手部模型]
---

# MANO

## 定义

MANO（**M**odel with **A**rticulated and **N**on-rigid def**O**rmation）是一种参数化手部网格模型，通过低维形状参数和姿态参数驱动手部网格变形，广泛用于手部姿态估计和重建。

## 数学形式

MANO 手部骨架等效运动链：

$$
\{T^{\text{MANO}}_{k,t}\}_k = \text{FK}(q^{\text{MANO}}_t, M^{\text{MANO}})
$$

$$
S^{\text{human}}_t = \text{Rasterise}\!\left(\left\{\pi\!\left(K_{\text{cam}},\; T^{\text{cam}}_{\text{world}}\, T^{\text{MANO}}_{k,t}\, o^{\text{MANO}}_k\right)\right\}_k,\; \mathcal{E}(M^{\text{MANO}})\right)
$$

## 核心要点

1. 21 个关节（16 个手指关节 + 手腕），通过 URDF 等效运动链建模
2. 形状参数（10 维）控制手的形态差异，姿态参数控制关节角度
3. 可从 RGB 视频或深度图中拟合，用于手部 mocap

## 代表工作

- [[OSCAR]]: 将 MANO 手部运动链与机器人运动链统一为相同格式的 2D 骨架，实现人-机器人跨本体训练

## 相关概念

- [[正向运动学]]
- [[骨架条件编码]]
- [[运动捕捉]]
