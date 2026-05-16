---
type: concept
aliases: [Cascaded World-Action-Model, 级联WAM, 级联世界动作模型]
---

# Cascaded WAM（级联世界动作模型）

## 定义
Cascaded WAM 是 WAM 的一种架构范式，通过顺序两阶段管线实现世界-动作映射：第一阶段生成未来状态的视觉规划，第二阶段从该规划中解码可执行的机器人命令。

## 数学形式

$$
p(o', a \mid o, l) = p(a \mid o', o, l) \cdot p(o' \mid o, l)
$$

## 核心要点
1. **显式规划（Explicit）**: 中间表示为像素空间视频帧；动作通过学习式逆动力学模型（IDM）或几何方法（光流、位姿跟踪）提取
2. **隐式规划（Implicit）**: 中间表示为潜变量序列，跳过像素解码降低延迟，动作直接从潜特征解码
3. **自然归纳偏置**: 世界模型无需推理机器人运动学，动作模型无需解决长时程场景预测

## 代表工作
- [[UniPi]]: 最早的 Cascaded WAM（文本条件视频生成 + IDM）
- [[WAM-Survey]]: Table 1 对 Cascaded WAM 方法的全面对比

## 相关概念
- [[WAM]]: 上层概念
- [[Inverse Dynamics Model]]: Cascaded WAM 第二阶段常用组件
- [[LDM]]: 隐式规划 Cascaded WAM 的关键技术
