---
type: concept
aliases: [Stanford Research Institute Problem Solver, STRIPS Planning, 命题规划]
---

# STRIPS

## 定义
STRIPS（Stanford Research Institute Problem Solver）是一种经典的 AI 规划形式语言，用命题符号表示世界状态、动作前提条件和效果，适合长时序自动规划。

## 数学形式

状态 $S$ 为命题集合，动作 $a$ 定义为三元组：

$$a = (\text{Pre}(a),\ \text{Add}(a),\ \text{Del}(a))$$

- $\text{Pre}(a)$：执行条件（前提命题集）
- $\text{Add}(a)$：执行后新增命题
- $\text{Del}(a)$：执行后删除命题

规划目标：找到动作序列 $\pi = (a_1, \ldots, a_n)$ 使 $S_n \supseteq G$（目标命题集）

## 核心要点
1. 世界状态用有限命题集合表示，丢弃无关视觉细节
2. 动作语义完全由 Add/Delete list 决定，可做符号推理
3. 与神经感知结合：用视觉编码器从图像中学习命题符号（如 [[FSQ]]）
4. 经典 SAT/BFS 规划器可直接在命题空间搜索

## 代表工作
- [[STRIPS-WM]]：从图像学习 STRIPS 风格的视觉世界模型，用于长时规划

## 相关概念
- [[CEM]]
- [[LeWM]]
