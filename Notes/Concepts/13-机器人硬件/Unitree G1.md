---
type: concept
aliases: [G1, Unitree G1 Humanoid]
---

# Unitree G1

## 定义
宇树科技（Unitree Robotics）开发的双足人形机器人，具有 23 个自由度，支持全身运动和简单操作，是目前人形机器人研究的主流平台之一。

## 核心要点

1. **自由度**: ~23 DOF（腿部 + 腰部 + 手臂）
2. **控制接口**: 关节位置 PD 控制，支持 500Hz 命令频率
3. **板载计算**: 搭载 Jetson Orin，支持 TensorRT 推理
4. **价格**: 相对低廉（约 16,000 USD），普及度高

## 在研究中的使用

- [[SONIC]]: 在 G1 上部署全身运动跟踪策略，99.2% 真实世界成功率
- 多项 VLA、全身控制研究的标准测试平台

## 相关概念
- [[Jetson Orin]]
- [[运动跟踪]]
- [[Isaac Lab]]
