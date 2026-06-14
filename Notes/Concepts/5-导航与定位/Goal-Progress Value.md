---
type: concept
aliases: [Goal-Progress Value, 目标进度值, 目标进度标量]
---

# Goal-Progress Value

## 定义
一种归一化到 $[0, 1]$ 的标量信号，度量机器人当前位置距导航目标终点的接近程度，用于辅助导航策略的监督训练。

## 数学形式

$$
v_{t+H} = \text{clip}\!\left(1 - \frac{\|p_\text{end} - p_t\|_2}{d_\text{max}},\; 0,\; 1\right)
$$

其中 $p_\text{end}$ 为轨迹终点，$p_t$ 为当前位置，$d_\text{max}$ 为轨迹全程距离。

## 核心要点
1. 值为 1 表示已到达目标，值为 0 表示处于轨迹起点
2. 在 NavWAM 中作为额外监督头，以"价值帧"形式编码到九帧潜在画布的第 8 帧
3. 消融实验证明加入该监督可显著降低 ATE（从 0.107 降至 0.076）

## 代表工作
- [[NavWAM]]: 首次在导航世界行动模型中引入目标进度值作为联合监督信号

## 相关概念
- [[Navigation World Model]]
- [[World-Action Model]]
