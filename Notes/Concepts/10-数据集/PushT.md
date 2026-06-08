---
type: concept
aliases: [Push-T, PushT Task]
---

# PushT

## 定义
PushT：一个标准的二维机器人推物体任务环境，智能体需将一个 T 形物体推到目标位置。因其可控性强、动力学非线性，成为 world model 和模仿学习研究的标准 benchmark。

## 数学形式
状态 $s = (x_{\text{agent}}, x_{\text{T}}, \theta_{\text{T}})$，动作 $a \in \mathbb{R}^2$（二维推力）。
成功率以 T 形物体与目标区域的覆盖比例 $\geq 0.95$ 为标准。

## 核心要点
1. 非线性接触动力学：推 T 形物体时会产生旋转，比推方形更难预测
2. 常用于评估 world model 的预测精度和 MPC 规划能力
3. [[SWM]] 将 PushT 纳入标准环境，支持颜色/物理属性的可控变异
4. 在零样本鲁棒性研究中，PushT 是最常见的测试床之一

## 代表工作
- [[stable-worldmodel]]: 用 PushT 评估 DINO-WM 的零样本跨分布泛化能力

## 相关概念
- [[SWM]]
- [[MPC]]
- [[MuJoCo]]
