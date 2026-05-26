---
type: concept
aliases: [Isaac Gym, IsaacGym, NVIDIA Isaac Gym]
---

# Isaac Gym

## 定义
NVIDIA 发布的高性能 GPU 加速物理仿真平台，支持并行大规模机器人强化学习与策略训练，X-DiffVLA 在此环境中验证跨具身抓取任务。

## 核心要点
1. **GPU 加速**: 支持数千个并行仿真环境，显著提升强化学习的采样效率
2. **机器人任务**: 支持灵巧手抓取、操作等高 DoF 任务仿真
3. **X-DiffVLA 使用**: 在 10 类物体、3 种末端执行器（Panda/Inspire/Shadow）上验证跨具身迁移

## 数学形式
无特定数学公式，但物理仿真采用刚体动力学引擎 PhysX。

## 代表工作
- [[X-DiffVLA]]: Isaac Gym 实验中提升 12.5%（vs π₀.₅）
- GR00T N1, OpenPI, 等多个 VLA 工作使用 Isaac Gym 验证策略

## 相关概念
- [[RoboCasa]]
- [[X-DiffVLA]]
