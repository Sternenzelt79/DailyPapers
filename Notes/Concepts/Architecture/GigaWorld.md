---
type: concept
aliases: [GigaWorld]
---

# GigaWorld

## 定义
GigaWorld：大规模视频生成式世界模型，用于模拟机器人或具身智能体与环境交互的动态过程，支持 policy 训练和规划。

## 数学形式
$$\hat{o}_{t+1:t+H} = f_\theta(o_{t-k:t}, a_{t:t+H-1})$$

## 核心要点
1. 以过去观测和动作序列为条件，预测未来帧序列（视频预测形式的世界模型）
2. 用于为 VLA/RL 策略提供 imagined rollout，减少真实环境交互需求
3. 大规模训练数据（"Giga" 级别）是核心，视频质量高度依赖数据覆盖

## 代表工作
- GigaWorld 系列工作（具体 arXiv 号待查）
- 同类工作：[[UniPi]]、[[DINO-WM]]、[[World Model]]

## 相关概念
- [[World Model]]
- [[UniPi]]
- [[VLA]]
