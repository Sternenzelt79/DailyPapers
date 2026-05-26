---
type: concept
aliases: [OpenAI Gym, gym, gymnasium]
---

# Gymnasium

## 定义

强化学习和控制研究的标准环境接口库（OpenAI Gym 的继任者），提供统一的 `reset()`/`step()` API，覆盖经典控制、Atari、MuJoCo 等数百个环境。

## 核心要点

1. 统一 `env.step(action) → obs, reward, terminated, truncated, info` 接口
2. 社区标准：绝大多数 RL/控制论文以 Gymnasium 兼容性为基础
3. 支持向量化并行环境（`VectorEnv`），加速数据收集
4. 由 Farama Foundation 维护，前身为 OpenAI Gym

## 代表工作

- [[StableWorldModel]]: swm 的环境包装器基于 Gymnasium 接口构建

## 相关概念

- [[MuJoCo]]
- [[World Model]]
- [[Model Predictive Control]]
