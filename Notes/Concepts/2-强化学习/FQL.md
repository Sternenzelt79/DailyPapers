---
type: concept
aliases: [Flow Q-Learning, FQL]
---

# FQL

## 定义
FQL（Flow Q-Learning）将 flow matching 与 Q-learning 结合，用 flow model 作为策略，通过 Q 函数梯度引导 flow 向高价值动作漂移，实现 offline RL 策略提升。

## 核心要点
1. 策略用 conditional flow model 参数化（可多步 NFE 控制）
2. Q-gradient 通过 ODE 路径反传，引导流场方向
3. 相比 DDPO，不依赖 reward fine-tuning，offline 数据友好

## 代表工作
- FQL (2024), Flow Q-Learning for Offline Reinforcement Learning

## 相关概念
- [[AWR]]
- [[IDQL]]
- [[Flow Matching]]
