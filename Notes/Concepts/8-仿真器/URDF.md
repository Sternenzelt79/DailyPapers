---
type: concept
aliases: [Unified Robot Description Format, 统一机器人描述格式]
---

# URDF

## 定义
Unified Robot Description Format，ROS 生态中用于描述机器人模型结构的 XML 格式标准，包含关节类型、连杆几何、质量惯量、碰撞形状等物理属性，是机器人仿真和控制的通用接口。

## 数学形式
关节变换（以 revolute joint 为例）：
$$T_{child}^{parent} = T_{origin} \cdot R(\theta, \hat{a})$$

其中 $\hat{a}$ 为关节轴，$\theta$ 为关节角。

## 核心要点
1. XML 格式，描述 link（连杆）和 joint（关节）的树状结构
2. 每个 link 包含：visual（渲染形状）、collision（碰撞形状）、inertial（质量/惯量）
3. 关节类型：fixed、revolute、prismatic、continuous、floating、planar
4. 被 MuJoCo、IsaacLab、Gazebo、PyBullet 等主流仿真器支持（通常需转换）

## 代表工作
- [[PhysX-Omni]]: 自动为铰接物体生成 URDF 格式输出，可直接导入 ROS/仿真器
- [[iMaC]]: 利用 URDF + 正向运动学将机器人动作渲染为 Motion Images，作为 world model 的视觉控制信号

## 相关概念
- [[MuJoCo]]
- [[IsaacLab]]
