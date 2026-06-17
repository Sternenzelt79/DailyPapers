---
type: concept
aliases: [Franka Emika Panda, Panda Robot, Franka Panda]
---

# Franka Emika Panda

## 定义
Franka Emika 公司出品的 7 自由度研究级机器人手臂，配备内置力矩传感器，是学术机器人操作研究中最广泛使用的平台之一。

## 核心要点
1. **自由度**: 7 个旋转关节 + 夹爪（共 8-DOF 配置）
2. **力矩传感**: 每个关节内置扭矩传感器，支持精确接触力控制
3. **控制接口**: libfranka 提供关节级控制，支持笛卡尔空间阻抗控制
4. **常用传感器配套**: 外部 RGB-D 摄像头（如 Zed 2i）+ 腕部摄像头（如 Zed Mini）

## 代表工作
- [[WEAVER]]: 使用 Panda + 2 个 Zed 2i + 1 个 Zed Mini 进行操作实验
- DROID 数据集：基于 Franka 采集的大规模操作数据

## 相关概念
- [[DROID]]
- [[pi0.5]]
