---
type: concept
aliases: [MuZero算法, Muzero]
---

# MuZero

## 定义
DeepMind 提出的无需环境模型的 MCTS 规划算法，通过学习潜在动态模型，在无需访问环境规则的情况下进行树搜索规划。

## 数学形式
$$v_k, r_k, s_{k+1} = g_\theta(s_k, a_k), \quad \pi_k, v_k = f_\theta(s_k)$$

其中 $g_\theta$ 为动态模型，$f_\theta$ 为预测网络，均在潜在空间中操作。

## 核心要点
1. 学习潜在状态转移模型（不是真实像素）：比 AlphaZero 更通用
2. 用 MCTS 在学习的潜在空间中规划，而非真实环境
3. 通过 value/policy/reward 三个头的联合监督训练
4. 在 Atari 和棋类游戏上超越所有 model-free 基线

## 代表工作
- [[MuZero]]: Schrittwieser et al., 2020，Nature
- [[UniZero]]: 在 MuZero 框架上统一了 representation learning
- [[COMET]]: 在 MuZero 风格 MCTS 上加入对象中心表征

## 相关概念
- [[MCTS]]
- [[UniZero]]
- [[MBRL]]
