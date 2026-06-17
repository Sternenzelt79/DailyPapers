---
type: concept
aliases: [接触图像, Contact Image, 接触控制图]
---

# Contact Images

## 定义

一种为 [[世界模型|Embodied World Models]] 设计的双流几何控制信号，将机器人夹爪与场景之间的空间接触关系编码为图像形式，替代低维动作向量，用于接触感知视频预测。

## 数学形式

**Robot-to-Scene 距离流**（机器人关键点到最近场景点的距离）：

$$
d^{r \to s}_{t+k}(r) = \min_{p \in P^s_t} \|r - p\|_2, \quad r \in P^r_{t+k}
$$

**Scene-to-Gripper 距离流**（场景点到最近夹爪点的距离）：

$$
d^{s \to g}_{t+k}(p) = \min_{g \in P^g_{t+k}} \|p - g\|_2, \quad p \in P^s_t
$$

两个距离场均以彩色深度图形式编码后注入视频生成 backbone。

## 核心要点

1. 通过几何距离场显式编码夹爪与目标物体的接近程度，解决接触时刻预测难题
2. 与 [[Motion Images]] 配合，提供运动 + 接触的完整控制信号
3. 依赖 [[Depth Anything 3]] 估计的场景深度图构建点云 $P^s_t$
4. 以图像形式与视频 backbone 天然兼容，无需特殊适配

## 代表工作

- [[iMaC]]: 提出 Contact Images 概念，在 8 个真实操作任务上验证有效性

## 相关概念

- [[Motion Images]]
- [[URDF]]
- [[正向运动学]]
- [[世界模型]]
