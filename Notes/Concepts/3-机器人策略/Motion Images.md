---
type: concept
aliases: [运动图像, Motion Image, 运动控制图]
---

# Motion Images

## 定义

一种将机器人动作序列转换为渲染图像的视觉控制表示，利用 [[URDF]] 和 [[正向运动学]] 直接计算未来机器人外观并渲染为图像，替代低维动作向量为 [[世界模型|Embodied World Models]] 提供像素级运动先验。

## 数学形式

三步生成管线：

$$
q_{t+k} = \phi(q_t, a_{t:t+k-1})
$$

$$
M_{t+k} = K_{\text{URDF}}(q_{t+k})
$$

$$
C^m_{t+k,v} = R(M_{t+k};\ K_v, T^v_{t+k})
$$

- $\phi$：关节积分函数
- $K_{\text{URDF}}$：正向运动学函数
- $R$：可微渲染器
- $K_v, T^v_{t+k}$：视角 $v$ 的相机内外参

## 核心要点

1. 将动作的运动语义"可视化"，让 world model 直接"看到"机器人未来的外观
2. 提供像素级空间对齐的运动引导，比向量表示信息量大得多
3. 依赖精确的机器人 URDF 模型和相机标定参数
4. 与 [[Contact Images]] 共同构成 iMaC 的双控制信号

## 代表工作

- [[iMaC]]: 提出 Motion Images，结合 Contact Images 实现接触感知 Embodied World Model

## 相关概念

- [[Contact Images]]
- [[URDF]]
- [[正向运动学]]
- [[世界模型]]
