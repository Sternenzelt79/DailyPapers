---
type: concept
aliases: [Franka Emika Panda, Panda机械臂, Franka机械臂]
---

# Franka Panda

## 定义
由 Franka Emika 开发的 7 自由度研究型机械臂，因其精确力控、开放 API 和适中价格，成为机器人操作研究中最广泛使用的平台之一。

## 数学形式

关节空间动作表示（10D 单臂 / 20D 双臂）：

$$
a = [q_1, q_2, q_3, q_4, q_5, q_6, q_7, \dot{q}_1, \dot{q}_2, g] \in \mathbb{R}^{10}
$$

其中 $q_i$ 为第 $i$ 关节角度，$g$ 为夹爪开度。

## 核心要点
1. **7 DoF**: 7 个旋转关节，冗余度高，灵活性强
2. **力控支持**: 内置关节力矩传感器，支持阻抗控制
3. **FCI 接口**: 1kHz 实时控制接口，支持直接关节控制
4. **研究生态**: 配套 franka_ros、robosuite、Deoxys 等多个控制框架

## 代表工作
- [[Cosmos3]]: 支持 Franka Panda 单臂（10D）和双臂（20D）的动作表示

## 相关概念
- [[前向动力学模型]]
- [[逆向动力学模型]]
- [[扩散策略]]
