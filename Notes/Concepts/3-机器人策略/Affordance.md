---
type: concept
aliases: [可供性, Affordance Map, 交互可供性, Affordance Forecasting]
---

# Affordance（可供性）

## 定义

Affordance 指环境或物体对智能体提供的**可能交互机会**，描述"哪个物体在哪里可以被如何操作"的结构化空间信息，是连接视觉感知与机器人动作生成的中间语义表示。

## 数学形式

二维 affordance 热力图（Where2Act 形式）：

$$
M \in \{0, 1\}^{H \times W}, \quad \hat{M} = \sigma(\text{Transformer\_Decoder}(v, l))
$$

其中 $v$ 为视觉特征，$l$ 为语言指令，$\sigma$ 为 Sigmoid 激活。

## 核心要点

1. **三层结构化分解**（Which / Where / How）：明确操作对象 → 精确交互位置 → 3D 执行几何
2. **任务导向中间表示**：affordance 将高层语义理解与低层运动控制解耦，缓解 VLA 表示坍塌
3. **2D 热力图**：基于像素的概率分布，标注人 / 机器人最可能的接触区域
4. **3D 几何扩展**：结合点云 / 深度信息，提供接触面法向量与空间布局约束

## 代表工作

- [[AffordanceVLA]]: 将结构化 affordance（Which2Act/Where2Act/How2Act）融入 VLA 框架，实现 affordance 感知的机器人操作
- Where2Act (ICCV 2021): 首个基于 affordance 热力图的机器人交互点预测方法

## 相关概念

- [[VLA]]: affordance 在 VLA 中作为中间表示桥接视觉语言理解与动作
- [[Mixture-of-Transformers]]: AffordanceVLA 中专门设计的 affordance 生成专家
- [[Diffusion Model]]: How2Act 3D 形状生成使用条件扩散
- [[Imitation Learning]]: affordance 监督可与模仿学习联合训练
