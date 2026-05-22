---
type: concept
aliases: [Temporal Difference Model Predictive Control 2, TD-MPC2]
---

# TD-MPC2

## 定义
结合时序差分学习（TD learning）与模型预测控制（MPC）的强化学习算法，通过联合训练世界模型、价值函数和奖励函数，实现高效在线连续控制。

## 核心要点
1. 在线 RL 训练，需要环境交互数据（与离线方法不同）
2. 在 DMControl Suite 任务上与 SAC 可比性能
3. **离线设置下严重失效**：stable-worldmodel 评测中 Push-T 仅 12%、OGBench 仅 4%，因为离线数据不包含 TD-MPC2 所需的探索性动作分布，导致动作漂移
4. 联合优化 $\{z_t, a_t, r_t, V_t\}$ 的潜在动态模型

## 局限性（stable-worldmodel 发现）
在离线机器人操作任务中，TD-MPC2 产生分布外动作生成，规划失效；说明在线 RL 方法不能直接应用于离线评测场景。

## 代表工作
- [[stable-worldmodel]]: 作为对比基线，揭示离线设置下的失效模式（2026）

## 相关概念
- [[模型预测控制]]
- [[Reinforcement Learning]]
- [[世界模型]]
