---
type: concept
aliases: [移动操作, Loco-Manipulation, 全身操作]
---

# Loco-Manipulation

## 定义

同时协调机器人下肢移动与上肢操作的全身控制任务，要求机器人在运动中完成物体抓取、放置等操作。

## 核心要点

1. 与纯操作任务不同，策略需同时处理平衡控制与末端执行器精度
2. 仿人机器人的 loco-manipulation 挑战尤为突出，需要全身协调
3. 训练数据难以通过传统遥操作规模化采集
4. 常见子任务：边走边取物、爬楼梯同时保持姿态、全身搬运

## 代表工作

- [[GRAIL]]: 通过全数字化流水线生成 2 万+ 条仿人 loco-manipulation 示范，在真实 Unitree G1 上达到 84% 取物成功率
- [[SONIC]]: 预训练全身运动控制器，为 loco-manipulation 提供运动先验

## 相关概念

- [[SONIC]]
- [[HOI]]
- [[PPO]]
