---
type: concept
aliases: [DiffAero, Diffusion for Aerial Robotics]
---

# DiffAero

## 定义
DiffAero 是将扩散模型应用于无人机轨迹规划和飞行控制的方法，通过扩散过程生成平滑且可行的飞行轨迹。

## 核心要点
1. 将轨迹生成建模为去噪扩散过程
2. 结合物理约束做 guided diffusion
3. 相比传统 MPC，能更好处理多模态轨迹分布

## 代表工作
- DiffAero (2024), Diffusion-based Agile Drone Flight

## 相关概念
- [[DreamerV3]]
- [[MAD]]
- [[DDPM]]
